import numpy as np
import os
from multiprocessing import Process, Pipe
import cloudpickle 


class CloudpickleWrapper(object):
    """
    使用 cloudpickle 来序列化内容，以便在多进程间传递复杂的 Python 对象，
    例如局部定义的函数或 lambda 函数。
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        """在序列化时调用，使用 cloudpickle.dumps()。"""
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        """在反序列化时调用，使用 cloudpickle.loads()。"""
        self.x = cloudpickle.loads(ob) # 反序列化也用 cloudpickle


def graphworker(remote, parent_remote, env_fn_wrapper):
    """
    在子进程中运行的环境工作函数

    Args:
        remote (multiprocessing.Connection): 子进程端的管道，用于与主进程通信。
        parent_remote (multiprocessing.Connection): 主进程端的管道 (在子进程中应关闭)。
        env_fn_wrapper (CloudpickleWrapper): 包装了创建环境实例的函数。
    """

    parent_remote.close()
    try:
        env = env_fn_wrapper.x()
        print(f"[DEBUG graphworker] 子进程 {os.getpid()}: 环境实例创建成功。")
    except Exception as e:
        print(f"[ERROR graphworker] 子进程 {os.getpid()}: 环境创建失败! Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            remote.send(("error_in_worker", str(e)))
        except Exception:
            pass
        return

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, ag_id, node_ob, adj, reward, done, info = env.step(data)
                remote.send((ob, ag_id, node_ob, adj, reward, done, info))
            elif cmd == "reset":
                ob, ag_id, node_ob, adj = env.reset()
                remote.send((ob, ag_id, node_ob, adj))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send(
                    (
                        env.observation_space,
                        env.share_observation_space,
                        env.action_space,
                        env.node_observation_space,
                        env.adj_observation_space,
                        env.edge_observation_space,
                        env.agent_id_observation_space,
                        env.share_agent_id_observation_space,
                    )
                )
            else:
                raise NotImplementedError
        except EOFError:
            break
        except Exception as e:
            print(f"GraphWorker Error: {e}")
            import traceback
            traceback.print_exc()
            try: # 尝试发送错误信号，避免主进程卡死
                remote.send(("error_in_worker_loop", str(e)))
            except Exception:
                pass
            break # 出错后子进程也应该退出循环

class GraphSubprocVecEnv:
    """
    用于图环境的、基于子进程的向量化环境，实现真正的并行数据采样。
    """
    def __init__(self, env_fns: list):
        """
        Args:
            env_fns (list): 包含创建单个环境实例的函数的列表。
        """
        self.waiting = False
        self.closed = False # 初始化 closed 状态
        nenvs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = []
        for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(
                target=graphworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            self.ps.append(process)
            process.daemon = True
            process.start()
        
        print(f"[DEBUG GraphSubprocVecEnv] 尝试启动的子进程数量: {len(self.ps)}")
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        spaces_tuple = self.remotes[0].recv()
        
        # 检查返回的是否是错误信号
        if isinstance(spaces_tuple, tuple) and spaces_tuple[0] == "error_in_worker":
            raise RuntimeError(f"Worker process failed to initialize or send spaces: {spaces_tuple[1]}")

        (
            observation_space, share_observation_space, action_space,
            node_observation_space, adj_observation_space, edge_observation_space,
            agent_id_observation_space, share_agent_id_observation_space,
        ) = spaces_tuple

        # 直接初始化原 ShareVecEnv 中的属性
        self.num_envs = nenvs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
        
        # 存储图环境特有的空间定义
        self.node_observation_space = node_observation_space
        self.adj_observation_space = adj_observation_space
        self.edge_observation_space = edge_observation_space
        self.agent_id_observation_space = agent_id_observation_space
        self.share_agent_id_observation_space = share_agent_id_observation_space

    def step_async(self, actions):
        assert len(actions) == self.num_envs, "动作数量必须与环境数量一致"
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = []
        for remote in self.remotes:
            result = remote.recv()
            # 检查子进程是否在循环中发送了错误信号
            if isinstance(result, tuple) and result[0] == "error_in_worker_loop":
                raise RuntimeError(f"Error in worker process during step: {result[1]}")
            results.append(result)
            
        self.waiting = False
        obs, ag_ids, node_obs, adj, rews, dones, infos = zip(*results)
        return (
            np.stack(obs), np.stack(ag_ids), np.stack(node_obs), np.stack(adj),
            np.stack(rews), np.stack(dones), infos,
        )

    def reset(self):
        # This reset is explicitly called by the main process (e.g., during warmup)
        for remote in self.remotes:
            remote.send(("reset", None))
        results = []
        for remote in self.remotes:
            result = remote.recv()
            # 检查子进程是否在循环中发送了错误信号 (虽然 reset 错误通常在创建时或 get_spaces 捕获)
            if isinstance(result, tuple) and result[0] == "error_in_worker_loop":
                raise RuntimeError(f"Error in worker process during reset: {result[1]}")
            results.append(result)
            
        obs, ag_ids, node_obs, adj = zip(*results)
        return (np.stack(obs), np.stack(ag_ids), np.stack(node_obs), np.stack(adj))

    def close_extras(self):
        """子类可以实现的额外清理工作 (如果需要，可以保留这个方法结构)。"""
        pass # GraphSubprocVecEnv 当前没有额外的清理

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    # 尝试接收任何挂起的消息，避免子进程在发送后阻塞
                    # 如果管道已关闭或另一端已退出，可能会抛出 EOFError 或 BrokenPipeError
                    remote.recv()
                except (EOFError, BrokenPipeError):
                    pass # 子进程可能已经关闭或管道损坏
                except Exception: # 其他可能的异常
                    pass

        # 调用 close_extras (如果它有实际操作的话)
        self.close_extras() # 保持与原 ShareVecEnv.close() 相似的结构

        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError: # 管道可能已经由于子进程退出而损坏
                pass
            except Exception:
                pass # 其他发送异常

        for p in self.ps:
            try:
                p.join(timeout=5) # 增加超时以应对潜在的清理延迟
                if p.is_alive():
                    print(f"Warning: Process {p.pid} did not terminate gracefully, forcing termination.")
                    p.terminate() # 如果超时仍未结束，则强制终止
                    p.join(timeout=1) # 等待强制终止完成
            except Exception as e:
                print(f"关闭子进程 {p.pid} 时出错: {e}")
        
        # 在所有进程处理完毕后，关闭主进程端的管道
        for remote in self.remotes:
            try:
                remote.close()
            except Exception: # 忽略关闭已关闭管道的错误
                pass

        self.closed = True

    def step(self, actions):
        """同步地执行环境步骤 (原 ShareVecEnv 中的方法)。"""
        self.step_async(actions)
        return self.step_wait()