import asyncio
import json
from asyncua import Client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- OPC UA 송신 설정 ---
# ⚠️ 주소 및 노드 ID는 수신 코드와 다르므로 주의하세요.
OPC_ENDPOINT = "opc.tcp://172.30.1.56:0630/freeopcua/server/"
OBJECT_NODE_ID = "ns=2;i=3"
METHOD_NODE_ID = "ns=2;i=25"

async def send_mission_state(client, status: str):
    """
    지정된 status 문자열을 JSON 형태로 변환하여 OPC UA 서버에 송신합니다.
    """
    mission_state = {
        "status": status
    }
    json_str = json.dumps(mission_state)
    
    # 노드 정보 가져오기
    obj = client.get_node(OBJECT_NODE_ID)
    method_node = client.get_node(METHOD_NODE_ID)
    
    print(f"\n[CALL] write_amr_mission_state(status='{status}')")
    
    # Call Method 실행
    result_code, result_message = await obj.call_method(
        method_node.nodeid,
        json_str # JSON 문자열 형태로 데이터 전달
    )
    
    print("  - ResultCode   :", result_code)
    print("  - ResultMessage:", result_message)
    logger.info(f"'{status}' 송신 완료.")

async def main():
    """
    메인 함수: OPC UA 서버에 연결하고 테스트 상태를 송신합니다.
    """
    # 초기 테스트 상태
    arm_mission_state = "PICK" 
    
    try:
        async with Client(OPC_ENDPOINT) as client:
            logger.info(f"[INFO] Connected to OPC UA Server: {OPC_ENDPOINT}")
            
            # 초기 상태 송신
            await send_mission_state(client, arm_mission_state)
            
            # 다른 상태 테스트
            await send_mission_state(client, "arm_mission_success")
            
            await asyncio.sleep(2.0)
            
    except Exception as e:
        logger.error(f"OPC UA 연결 또는 송신 중 오류 발생: {e.__traceback__}")

if __name__ == "__main__":
    asyncio.run(main())