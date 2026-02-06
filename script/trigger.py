import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import time
import sys

# 配置脚本路径映射
SCRIPT_MAP = {
    "/dycp": "./dycp.sh",
    "/dp": "./dp.sh",
    "/dcp": "./dcp.sh",
    "/hunbu": "./hunbu.sh"
}

PORT = 9999
LOG_STDOUT = "vllm_start_stdout.log"
LOG_STDERR = "vllm_start_stderr.log"

# 全局参数变量
GLOBAL_ARG = None

def kill_vllm():
    """彻底清理所有 vLLM 相关进程，包括主程序、协调器、引擎核心和工作进程"""
    print("\n" + "="*40)
    print("开始全方位清理 vLLM 进程...")
    
    # 1. 杀掉主 vllm 进程
    print("1/5: 杀掉 vllm 主进程...")
    subprocess.run("pkill -9 -f vllm 2>/dev/null || true", shell=True)
    time.sleep(3)
    
    # 2. 杀掉 DPCoordinator
    print("2/5: 杀掉 DPCoordinator (分布式协调器)...")
    subprocess.run("pkill -9 -f DPCoordinator 2>/dev/null || true", shell=True)
    time.sleep(3)
    
    # 3. 杀掉 EngineCore
    print("3/5: 杀掉 EngineCore (引擎核心)...")
    subprocess.run("pkill -9 -f EngineCore 2>/dev/null || true", shell=True)
    time.sleep(3)
    
    # 4. 杀掉 Worker (针对你发现的 VLLM::Worker_Domain...)
    print("4/5: 杀掉 VLLM Worker (工作进程)...")
    # 使用 -i 忽略大小写，并匹配 Worker 关键字
    subprocess.run("pkill -9 -f Worker 2>/dev/null || true", shell=True)
    # 针对性杀掉带 VLLM 标记的进程
    subprocess.run("pkill -9 -f VLLM 2>/dev/null || true", shell=True)
    time.sleep(3)
    
    # 5. 最终静默期
    print("5/5: 等待 2s 确保显存完全释放...")
    time.sleep(2)
    print("清理完成，环境已就绪。")
    print("="*40 + "\n")

class TriggerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in SCRIPT_MAP:
            script_path = SCRIPT_MAP[self.path]
            
            if not os.path.exists(script_path):
                self.send_error_msg(f"Error: Script {script_path} not found.")
                return

            try:
                # 启动前执行深度清理
                kill_vllm()

                print(f"Action: 正在执行脚本 {script_path} 参数: {GLOBAL_ARG}")
                
                # 构建执行命令
                cmd = ["/bin/bash", script_path]
                if GLOBAL_ARG:
                    cmd.append(GLOBAL_ARG)
                
                # 使用 append 模式记录日志
                with open(LOG_STDOUT, "a") as out, open(LOG_STDERR, "a") as err:
                    # 启动新的进程组
                    subprocess.Popen(
                        cmd,
                        stdout=out,
                        stderr=err,
                        start_new_session=True 
                    )

                self.send_response(200)
                self.end_headers()
                response_txt = f"Successfully cleaned environment and triggered {self.path} with arg: {GLOBAL_ARG}"
                self.wfile.write(response_txt.encode())

            except Exception as e:
                self.send_error_msg(str(e))
        
        elif self.path == '/stop':
            kill_vllm()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"All vLLM processes (including Workers) stopped.")
            
        else:
            self.send_response(404)
            self.end_headers()

    def send_error_msg(self, msg):
        self.send_response(500)
        self.end_headers()
        self.wfile.write(msg.encode())
        print(f"ERROR: {msg}")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        GLOBAL_ARG = sys.argv[1]
        print(f"设置全局参数: {GLOBAL_ARG}")
    else:
        print("未提供参数，脚本将不带参数执行")
    
    # 自动修正脚本执行权限
    for path in SCRIPT_MAP.values():
        if os.path.exists(path):
            os.chmod(path, 0o755)

    print(f"--- 远程触发服务端已启动 ---")
    print(f"监听端口: {PORT}")
    print(f"可用端点:")
    for path in SCRIPT_MAP.keys():
        print(f"  http://localhost:{PORT}{path}")
    print(f"  http://localhost:{PORT}/stop")
    print(f"日志文件: {LOG_STDOUT}")
    print(f"全局参数: {GLOBAL_ARG if GLOBAL_ARG else '无'}")
    
    server = HTTPServer(('0.0.0.0', PORT), TriggerHandler)
    server.serve_forever()
