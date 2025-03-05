import asyncio
import json
#from websockets.sync.client import connect
from  websockets.asyncio.client import connect

File_object = open(r"important.txt")
info = File_object.readlines()
key = info[0].replace("\n", "")
secret = info[1].replace("\n", "")
uri = "wss://stream.data.alpaca.markets/v2/test"

key_info = {"action": "auth",
                "key": key,
                "secret": secret}

stock_action = {"action": "subscribe",
            "trades": ["FAKEPACA"]}

async def chat():
    async with connect(uri) as websocket:
        #ensure websocket connection to uri
        connection_check = await websocket.recv()
        print(f"<<< {connection_check}")

        #authenticate
        print(f">>> {json.dumps(key_info)}")
        await websocket.send(json.dumps(key_info))
        auth_check = await websocket.recv()
        print(f"<<< {auth_check}")

        #subcribe
        print(f">>> {stock_action}")
        await websocket.send(json.dumps(stock_action))
        sub_confirm = await websocket.recv()
        print(f"<<< {sub_confirm}")

        while True:
            # Prompt the user for a message
            #message = input("Enter message: ")
            # Send the message to the server
            #await websocket.send(message)
            # Receive a message from the server
            response = await websocket.recv()
            print(f"Received: {response}")

if __name__ == "__main__":
    #stick_live_stream()
    asyncio.run(chat())
