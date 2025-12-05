import asyncio
import json
import time
from asyncua import Client, ua
# MyCobot 대신 MyCobot320을 사용할 수 있도록 수정
from pymycobot.mycobot import MyCobot
from pymycobot import MyCobot320 
import logging

# 로깅 설정 (옵션)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MyCobot 설정 ---
PORT = "COM3"
BAUD = 115200

# MyCobot 객체는 전역으로 선언하여 OPC UA 핸들러와 메인 함수에서 접근합니다.
# MyCobot320 객체를 사용하므로, mc는 MyCobot320 인스턴스가 됩니다.
mc = None

# --- OPC UA 수신/송신 설정 (주소는 원본 코드를 유지) ---
OPCUA_READ_URL = "opc.tcp://172.30.1.61:0630/freeopcua/server/" # 수신 서버 주소
OPCUA_WRITE_URL = "opc.tcp://172.30.1.61:0630/freeopcua/server/" # 송신 서버 주소

# OPCUA_READ_URL = "opc.tcp://172.30.1.56:0630/freeopcua/server/" # 수신 서버 주소
# OPCUA_WRITE_URL = "opc.tcp://172.30.1.56:0630/freeopcua/server/" # 송신 서버 주소

OBJECT_NODE_ID = "ns=2;i=3"
METHOD_NODE_ID = "ns=2;i=25"

# ----------------------
# 30_write_OPCUA.py 의 send_mission_state 함수 통합
# ----------------------
async def send_mission_state(status: str):
    """
    별도의 OPC UA 클라이언트를 생성하여 미션 상태(status)를 서버에 송신합니다.
    """
    mission_state = {
        "status": status
    }
    json_str = json.dumps(mission_state)
    
    logger.info(f"OPC UA 송신 서버에 연결 시도: {OPCUA_WRITE_URL}")
    try:
        # 송신 전용 클라이언트 생성 및 연결
        async with Client(OPCUA_WRITE_URL) as client:
            obj = client.get_node(OBJECT_NODE_ID)
            method_node = client.get_node(METHOD_NODE_ID)
            
            print(f"\n[OPC UA WRITE] write_amr_mission_state(status='{status}')")
            # OPC UA Call Method 실행
            result_code, result_message = await obj.call_method(
                method_node.nodeid,
                json_str
            )
            print("  - ResultCode   :", result_code)
            print("  - ResultMessage:", result_message)
            logger.info("OPC UA 송신 완료.")

    except Exception as e:
        logger.error(f"OPC UA 송신 중 오류 발생: {e}")
        logger.error("OPC UA 송신 중 치명적인 오류 발생", exc_info=True)

# ----------------------
# OPC UA DataChange 구독 핸들러 클래스
# ----------------------
class SubHandler:
    
    def __init__(self, mycobot_instance):
        self.mc = mycobot_instance
        logger.info("SubHandler 초기화 완료.")
        

    def datachange_notification(self, node, val, data):
        """
        데이터 변경 알림 시 호출되는 비동기적 콜백 함수
        """
        # 비동기 함수인 execute_command_and_respond를 별도의 태스크로 실행
        asyncio.create_task(self.execute_command_and_respond(val))

    async def execute_command_and_respond(self, val):
        """
        명령을 파싱하고 MyCobot 동작을 수행한 후 응답합니다.
        """
        
        # 1. 수신된 값 출력
        print(f"OPC UA 수신 값: {val}")

        # 2. JSON 파싱 시도
        command = None
        if isinstance(val, str):
            try:
                json_data = json.loads(val)
                logger.info(f"JSON 파싱 성공: {json_data}")
                
                if "move_command" in json_data:
                    command = json_data["move_command"]
                
            except json.JSONDecodeError:
                logger.warning(f"JSON 파싱 실패 (일반 문자열): {val}")
                command = val # Ready 같은 일반 문자열도 command로 간주

        # 3. MyCobot 동작 수행 및 응답
        if command and self.mc is not None:
            # MyCobot320을 사용하더라도 pymycobot API는 대부분 동일하게 동작합니다.
            if command == "go_home":
                logger.info("-> MyCobot: go_home 명령 수행 (로봇 홈 위치로 이동)")
                self.mc.send_angles([0, 0, 90, 0, -90, -90], 50)
            
            elif command == "mission_start":
                logger.info("-> MyCobot: mission_start 명령 수행 (미션 시작 위치/동작)")
                
                # --- 미션 시작 동작 ---
                mc.set_gripper_value(80, 50)
                time.sleep(3)
                self.mc.send_angles([0, 0, 0, 0, 0, 0], 50)
                time.sleep(3)
                # go home
                self.mc.send_angles([-17.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50)
                time.sleep(3)
                # pick
                self.mc.send_angles([-27.15, 7.55, 118.56, -31.72, -84.99, -119.35], 50)
                time.sleep(3)
                mc.set_gripper_value(25, 50)
                time.sleep(3)
                self.mc.send_angles([-17.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50)
                time.sleep(3)
                self.mc.send_angles([25.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50)
                time.sleep(3)
                # place
                self.mc.send_angles([30, 21.79, 68.11, -0.7, -80.41, -65.56], 50)
                time.sleep(3)
                mc.set_gripper_value(80, 50)
                time.sleep(3)
                self.mc.send_angles([-17.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50)
                time.sleep(3)
                self.mc.send_angles([0, 0, 0, 0, 0, 0], 50)
                time.sleep(3)
                # --- 미션 종료 ---
                
                # 동작 완료 대기 후 응답
                await asyncio.sleep(5) 
                
                logger.info("-> MyCobot: mission_start 동작 완료. OPC UA 응답 송신 시작.")
                # 4. OPC UA 송신 태스크 호출 (arm_mission_success)
                await send_mission_state("arm_mission_success")
            
            elif command == "Ready":
                logger.info("-> MyCobot: Ready 상태 수신, 대기 중...")
                
            else:
                logger.warning(f"-> MyCobot: 알 수 없는 명령: {command}")


async def amr_subscriber():
    """
    OPC UA 클라이언트를 실행하고 구독을 설정하는 메인 함수
    """
    global mc

    # MyCobot 연결 초기화
    try:
        # MyCobot320 객체 생성
        mc = MyCobot320(PORT, BAUD)
        mc.set_color(0, 0, 255) 
        logger.info(f"MyCobot320이 {PORT}에 {BAUD} 속도로 성공적으로 연결되었습니다.")
        
        # --- 추가된 그리퍼 초기화 로직 ---
        logger.info("-> MyCobot320: 전기 그리퍼 초기화 시작")
        mc.set_gripper_mode(0)          # 0: 전동 그리퍼 모드 설정
        mc.init_electric_gripper()      # 그리퍼 초기화
        time.sleep(2)
        mc.set_electric_gripper(0)      # 그리퍼 열림
        # 그리퍼 위치 설정 (value: 55, speed: 20, set_flag: 1)
        mc.set_gripper_value(55, 20, 1) 
        time.sleep(2)
        logger.info("-> MyCobot320: 전기 그리퍼 초기화 완료 (55 위치로 이동).")
        # --------------------------------
        
    except Exception as e:
        logger.error(f"MyCobot320 연결 또는 초기화 실패: {e}")
        mc = None

    if mc is None:
        logger.error("MyCobot 연결 문제로 OPC UA 구독을 시작할 수 없습니다. OPC UA 통신만 진행합니다.")

    logger.info(f"OPC UA 수신 서버에 연결 시도: {OPCUA_READ_URL}")
    try:
        async with Client(OPCUA_READ_URL) as client:
            logger.info("OPC UA 수신 서버에 성공적으로 연결되었습니다.")
            
            handler = SubHandler(mc)
            sub = await client.create_subscription(100, handler)
            
            # 구독할 노드 경로
            node_path = [
                "0:Objects",
                "2:ARM",
                "2:read_arm_go_move"
            ]
            cmd_node = await client.nodes.root.get_child(node_path)
            
            await sub.subscribe_data_change(cmd_node)
            logger.info(f"노드 '{node_path[-1]}' 구독 시작. 데이터 수신 대기 중...")

            while True:
                await asyncio.sleep(1) # 클라이언트 유지
    
    except Exception as e:
        logger.error(f"OPC UA 연결 또는 구독 중 오류 발생: {e}")
    finally:
        if mc is not None:
            mc.set_color(0, 0, 0)
            logger.info("OPC UA 클라이언트 종료. MyCobot 정리.")


if __name__ == "__main__":
    try:
        asyncio.run(amr_subscriber())
    except KeyboardInterrupt:
        logger.info("사용자 중단 (Ctrl+C). 프로그램 종료.")
    except Exception as e:
        logger.critical(f"프로그램 최종 오류: {e}")