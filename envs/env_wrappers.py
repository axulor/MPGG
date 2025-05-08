# env_wrappers.py (修改 CloudpickleWrapper)
import numpy as np
import os
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
# import pickle # 不再直接使用标准 pickle 进行序列化

# --- 恢复使用 cloudpickle ---
import cloudpickle # 导入 cloudpickle


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


class ShareVecEnv(ABC):
    """
    向量化环境的抽象基类。
    定义了与多个环境副本交互的统一接口。
    """
    closed = False
    # metadata = {"render.modes": ["human", "rgb_array"]} # 渲染相关，精简版可移除

    def __init__(self, num_envs: int, observation_space, share_observation_space, action_space):
        """
        初始化向量化环境基类。
        Args:
            num_envs (int): 包含的环境实例数量。
            observation_space: 单个环境的个体观测空间。
            share_observation_space: 单个环境的共享观测空间。
            action_space: 单个环境的动作空间。
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """重置所有环境并返回一批初始观测。"""
        pass

    @abstractmethod
    def step_async(self, actions):
        """异步地向所有环境发送动作指令。"""
        pass

    @abstractmethod
    def step_wait(self):
        """等待异步步骤完成并返回结果。"""
        pass

    def close_extras(self):
        """子类可以实现的额外清理工作。"""
        pass

    def close(self):
        """关闭所有环境并释放资源。"""
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """同步地执行环境步骤。"""
        self.step_async(actions)
        return self.step_wait()

# --- graphworker 函数 (GraphSubprocVecEnv 的核心) ---
def graphworker(remote, parent_remote, env_fn_wrapper):
    """
    在子进程中运行的环境工作函数。

    Args:
        remote (multiprocessing.Connection): 子进程端的管道，用于与主进程通信。
        parent_remote (multiprocessing.Connection): 主进程端的管道 (在子进程中应关闭)。
        env_fn_wrapper (CloudpickleWrapper): 包装了创建环境实例的函数。
    """

    parent_remote.close()
    try: # 添加 try-except 捕获环境创建时的错误
        env = env_fn_wrapper.x()
        print(f"[DEBUG graphworker] 子进程 {os.getpid()}: 环境实例创建成功。") # <--- 添加这行
    except Exception as e:
        print(f"[ERROR graphworker] 子进程 {os.getpid()}: 环境创建失败! Error: {e}")
        import traceback
        traceback.print_exc()
        # 可以选择通过 remote 发送一个错误信号给主进程，或者直接让子进程退出
        try:
            remote.send(("error_in_worker", str(e))) # 发送错误信息
        except Exception: # 如果管道也坏了
            pass
        return # 子进程退出
    # env = env_fn_wrapper.x() # 调用 CloudpickleWrapper 反序列化并获取函数，然后执行得到环境实例

    while True: # 命令接收和处理循环
        try:
            cmd, data = remote.recv() # 从主进程接收命令和数据
            if cmd == "step":
                # 执行一步环境模拟
                # **关键:** 返回值的顺序和数量必须与环境的 step() 方法一致
                # MultiAgentGraphEnv.step 返回: obs, agent_id, node_obs, adj, rewards, dones, infos
                ob, ag_id, node_ob, adj, reward, done, info = env.step(data)
                # 如果环境完成，则自动重置
                # np.all(done) 检查是否所有智能体都完成了 (对于多智能体环境)
                if np.all(done): # 假设 done 是 (num_agents,) 的布尔数组
                    # **关键:** 重置后也需要返回相同结构的数据
                    ob, ag_id, node_ob, adj = env.reset()
                remote.send((ob, ag_id, node_ob, adj, reward, done, info))
            elif cmd == "reset":
                # 重置环境
                # **关键:** 返回值的顺序和数量必须与环境的 reset() 方法一致
                # MultiAgentGraphEnv.reset 返回: obs, agent_id, node_obs, adj
                ob, ag_id, node_ob, adj = env.reset()
                remote.send((ob, ag_id, node_ob, adj))
            elif cmd == "close":
                # 关闭环境并退出循环
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                # 主进程请求获取环境的空间定义
                # **关键:** 返回的空间顺序和数量必须与 GraphSubprocVecEnv 初始化时期望的一致
                remote.send(
                    (
                        env.observation_space,       # 个体观测空间
                        env.share_observation_space, # 共享观测空间
                        env.action_space,            # 动作空间
                        # 图环境特有的空间
                        env.node_observation_space,
                        env.adj_observation_space,
                        env.edge_observation_space,     # 边特征空间
                        env.agent_id_observation_space,
                        env.share_agent_id_observation_space,
                    )
                )
            else:
                raise NotImplementedError # 未知命令
        except EOFError: # 如果主进程关闭了管道，则子进程退出
            break
        except Exception as e: # 捕获其他潜在错误并打印
            print(f"GraphWorker Error: {e}")
            import traceback
            traceback.print_exc()
            remote.send(None) # 发送一个信号表示出错，避免主进程卡死
            break


class GraphDummyVecEnv(ShareVecEnv):
    """
    用于图环境的“假的”向量化环境 (在主进程中串行运行)。
    主要用于单线程训练/评估，或调试。
    """
    def __init__(self, env_fns: list): # env_fns 是一个包含创建环境的函数的列表
        """
        Args:
            env_fns (list): 包含创建单个环境实例的函数的列表。
                            例如: [lambda: MultiAgentGraphEnv(cfg)]
        """
        # 调用列表中的每个函数来创建环境实例
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0] # 取第一个环境作为代表，获取空间信息

        # 初始化父类 ShareVecEnv
        super(GraphDummyVecEnv, self).__init__(
            num_envs=len(env_fns),
            observation_space=env.observation_space,
            share_observation_space=env.share_observation_space,
            action_space=env.action_space,
        )
        self.actions = None # 用于临时存储待执行的动作

        # 存储图环境特有的空间定义
        # 这些属性允许 Runner 在初始化 Policy 和 Buffer 时访问它们
        self.node_observation_space = env.node_observation_space
        self.adj_observation_space = env.adj_observation_space
        self.edge_observation_space = env.edge_observation_space # 边特征空间
        self.agent_id_observation_space = env.agent_id_observation_space
        self.share_agent_id_observation_space = env.share_agent_id_observation_space

    def step_async(self, actions):
        """在 DummyVecEnv 中，异步步骤只是简单地存储动作。"""
        self.actions = actions

    def step_wait(self):
        """串行执行存储的动作，并收集所有环境的结果。"""
        results = []
        # 遍历每个环境和对应的动作
        for i, (action, env) in enumerate(zip(self.actions, self.envs)):
            # 执行一步
            # 返回值顺序和数量需与 env.step() 匹配
            obs_one, ag_id_one, node_obs_one, adj_one, reward_one, done_one, info_one = env.step(action)
            # 如果环境完成，则自动重置
            if np.all(done_one): # 假设 done_one 是 (num_agents,)
                # 重置后也需要返回相同结构的数据
                obs_one, ag_id_one, node_obs_one, adj_one = env.reset()
            results.append((obs_one, ag_id_one, node_obs_one, adj_one, reward_one, done_one, info_one))

        # 将所有环境的结果解包并使用 np.array 堆叠成批处理形式
        # (如果只有一个环境，np.array 也能正常工作，增加一个批处理维度)
        obs, ag_ids, node_obs, adj, rews, dones, infos = map(np.array, zip(*results))

        self.actions = None # 清空已执行的动作
        return obs, ag_ids, node_obs, adj, rews, dones, infos # 返回批量的结果

    def reset(self):
        """重置所有环境实例，并返回批量的初始观测。"""
        results = []
        for env in self.envs:
            # 返回值顺序和数量需与 env.reset() 匹配
            obs_one, ag_id_one, node_obs_one, adj_one = env.reset()
            results.append((obs_one, ag_id_one, node_obs_one, adj_one))
        # 解包并堆叠
        obs, ag_id, node_obs, adj = map(np.array, zip(*results))
        return obs, ag_id, node_obs, adj

    def close(self):
        """关闭所有环境实例。"""
        for env in self.envs:
            env.close()


class GraphSubprocVecEnv(ShareVecEnv):
    """
    用于图环境的、基于子进程的向量化环境，实现真正的并行数据采样。
    """
    def __init__(self, env_fns: list):
        """
        Args:
            env_fns (list): 包含创建单个环境实例的函数的列表。
        """
        self.waiting = False # 标记是否正在等待子进程返回结果
        self.closed = False
        nenvs = len(env_fns) # 环境数量

        # 创建管道 (Pipe) 用于主进程和子进程之间的双向通信
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # 创建并启动子进程
        self.ps = []
        for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns):
            # 每个子进程的目标是 graphworker 函数
            process = Process(
                target=graphworker, # 子进程执行的函数
                args=(work_remote, remote, CloudpickleWrapper(env_fn)), # 传递给 graphworker 的参数
            )
            self.ps.append(process)
            process.daemon = True # 设置为守护进程
            process.start() # 启动子进程
        
        print(f"[DEBUG GraphSubprocVecEnv] 尝试启动的子进程数量: {len(self.ps)}")
        # 关闭主进程中不再需要的子进程端管道连接
        for remote in self.work_remotes:
            remote.close()

        # 从第一个子进程获取空间信息 (假设所有环境空间相同)
        self.remotes[0].send(("get_spaces", None)) # 发送 "get_spaces" 命令
        (
            observation_space, share_observation_space, action_space,
            node_observation_space, adj_observation_space, edge_observation_space,
            agent_id_observation_space, share_agent_id_observation_space,
        ) = self.remotes[0].recv() # 接收返回的空间信息

        # 初始化父类 ShareVecEnv
        super(GraphSubprocVecEnv, self).__init__(
            num_envs=nenvs,
            observation_space=observation_space,
            share_observation_space=share_observation_space,
            action_space=action_space,
        )
        # 存储图环境特有的空间定义
        self.node_observation_space = node_observation_space
        self.adj_observation_space = adj_observation_space
        self.edge_observation_space = edge_observation_space
        self.agent_id_observation_space = agent_id_observation_space
        self.share_agent_id_observation_space = share_agent_id_observation_space

    def step_async(self, actions):
        """异步地将动作发送给所有子进程。"""
        assert len(actions) == self.num_envs, "动作数量必须与环境数量一致"
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action)) # 发送 "step" 命令和对应的动作
        self.waiting = True # 标记为正在等待结果

    def step_wait(self):
        """等待所有子进程完成步骤并收集结果。"""
        results = [remote.recv() for remote in self.remotes] # 从每个子进程接收结果
        self.waiting = False
        # 解包结果
        # obs, ag_ids, node_obs, adj, rews, dones, infos
        obs, ag_ids, node_obs, adj, rews, dones, infos = zip(*results)
        # 将结果堆叠成批处理形式的 NumPy 数组
        return (
            np.stack(obs), np.stack(ag_ids), np.stack(node_obs), np.stack(adj),
            np.stack(rews), np.stack(dones), infos,
        )

    def reset(self):
        """重置所有子进程中的环境。"""
        for remote in self.remotes:
            remote.send(("reset", None)) # 发送 "reset" 命令
        results = [remote.recv() for remote in self.remotes] # 接收结果
        obs, ag_ids, node_obs, adj = zip(*results) # 解包
        # 堆叠成批处理形式
        return (np.stack(obs), np.stack(ag_ids), np.stack(node_obs), np.stack(adj))

    def close(self):
        """关闭所有子进程和管道。"""
        if self.closed:
            return
        if self.waiting: # 如果还在等待异步步骤完成，则先接收结果
            for remote in self.remotes:
                try:
                    remote.recv() # 尝试接收，避免阻塞
                except EOFError:
                    pass # 如果管道已关闭则忽略
        # 向所有子进程发送 "close" 命令
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError: # 如果管道已损坏则忽略
                pass
        # 等待所有子进程结束
        for p in self.ps:
            try:
                p.join(timeout=1) # 设置超时，避免无限等待
                if p.is_alive():
                    p.terminate() # 如果超时仍未结束，则强制终止
                    p.join()
            except Exception as e:
                print(f"关闭子进程时出错: {e}")
        self.closed = True