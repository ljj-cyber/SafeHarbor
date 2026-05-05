import os
import json
import importlib
from inspect import signature


class Env:
    """
    环境类，用于存储工具描述和调用工具函数。
    """
    def __init__(self, tools=None, env_name=None, parameters=None):
        """
        初始化环境，传入工具描述和环境名称。
        :param tools: 所有工具列表
        :param env_name: 环境名称（对应模块名）
        :param parameters: 环境参数
        """
        self.tools = tools
        self.env_name = env_name
        self.parameters = parameters

    def has_tool(self, tool_name):
        """
        检查环境是否支持指定的工具。
        :param tool_name: 工具名称
        :return: 如果工具存在，返回True；否则返回False
        """
        return tool_name in self.tools

    def get_tool_descs(self, tool_names):
        """
        根据工具名称返回工具描述。
        :param tool_names: 工具名称列表
        :return: 返回工具描述的列表
        """
        tool_descs = []
        for tool_name in tool_names:
            tool_desc = self.tools.get(tool_name, None)
            if tool_desc:
                tool_descs.append(tool_desc)
            else:
                print(f"Warning: Tool {tool_name} not found in the environment.")
        return tool_descs

    def call_tool(self, tool_name, arguments):
        """
        根据工具名称调用相应的工具。
        :param tool_name: 工具名称
        :param arguments: 工具参数
        :return: 工具执行结果
        """
        tool_desc = self.tools.get
        if tool_desc is None:
            return {"success": False, "message": f"Tool {tool_name} not found."}

        # 动态加载模块，环境名称对应模块名
        try:
            env_module = importlib.import_module(self.env_name)
            if hasattr(env_module, self.env_name):  # 检查模块中是否包含环境名对应的类
                env_class = getattr(env_module, self.env_name)  # 获取类
                # 实例化类，传递 parameters
                env_instance = env_class(parameters=self.parameters)

                if hasattr(env_instance, tool_name):  # 检查类中是否包含工具方法
                    tool_function = getattr(env_instance, tool_name)  # 获取工具函数
                    try:
                        # 确保参数与工具函数签名匹配
                        sig = signature(tool_function)
                        required_params = sig.parameters
                        print(arguments)
                        for param in required_params:
                            if param not in arguments:
                                raise ValueError(f"Missing required argument: {param}")

                        # 调用工具函数
                        result = tool_function(**arguments)
                        return {"success": True, "result": result}
                    except Exception as e:
                        return {"success": False, "message": str(e)}
                else:
                    return {"success": False, "message": f"Tool {tool_name} not found in class {self.env_name}."}
        except ModuleNotFoundError:
            return {"success": False, "message": f"Module {self.env_name} not found."}


class EnvManager:
    def __init__(self, environments_dir='../environments'):
        self.environments_dir = environments_dir
        self.tools = {}  # 存储所有要加载的工具 (key: name, value: description)
        self.envs = {}    # 存储环境
        self.load_tools()

    def load_tools(self):
        """
        加载environments目录下的所有JSON文件，并根据描述注册工具。
        """
        for filename in os.listdir(self.environments_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.environments_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        tool_descs = json.load(f)  # 读取 JSON 数据
                        # 如果是一个列表格式，逐个处理每个工具描述
                        if isinstance(tool_descs, list):
                            for tool_desc in tool_descs:
                                tool_name = tool_desc['name']
                                self.tools[tool_name] = tool_desc
                        else:
                            # 如果是字典格式，直接处理
                            tool_name = tool_descs['name']
                            self.tools[tool_name] = tool_descs
                except Exception as e:
                    print(f"Error loading tool description from {filepath}: {e}")


    def init_env(self, env_name, env_params=None):
        """
        初始化环境
        """
        # 获取该环境的工具描述，并传递 parameters
        env = Env(tools=self.tools, env_name=env_name, parameters=env_params)
        self.envs[env_name] = env
        return env

    def get_tool_descs(self, tool_names):
        """
        获取指定工具名称的工具描述
        """
        tool_descs = []
        for tool_name in tool_names:
            if tool_name in self.tools:
                tool_descs.append(self.tools[tool_name])
            else:
                print(f"Tool {tool_name} not found.")
        return tool_descs


    def has_tool(self, env_name, tool_name):
        """
        检查指定环境是否支持某个工具。
        """
        if env_name not in self.envs:
            raise ValueError(f"Environment {env_name} not found.")
        
        env = self.envs[env_name]
        return env.has_tool(tool_name)
    

if __name__ == '__main__':
    pass