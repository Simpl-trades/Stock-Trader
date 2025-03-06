import asyncio
import json
from  websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

'''
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

'''
class WebSocketClient:
    #instance will act as a flag to ensure that this is a single instance of object
    #lock ensures thread safety handling
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        '''
        defining use of new keyword for WSC class
        :param 1: class 
        :param 2: args for cls init
        :param 3: kwargs for cls init
        '''
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, uri: str, key: str, secret: str, callback=None):
        if not hasattr(self, "uri"):    # prevent reinit
            self.uri = uri
            self.key = key
            self.secret = secret
            self.websocket = None
            self.connected = False
            self.callback = callback # function to process recieved data
    
    async def send(self, message: dict):
        '''
        :pre: instance + string to be sent
            note: for our use cases we will largely send json msgs
        :post: sent message
        :param 1: dictionary with message for alpaca
        :returns: void
        '''
        if self.connected and self.websocket:
            await self.websocket.send(json.dumps(message))
            print(f">>> {message}")
        else:
            print("WebSocket is not connected!")

    async def recieve(self):
        '''
        :pre: instance up
        :post: recieve message from websocket
        '''
        if self.connected and self.websocket:
            response = await self.websocket.recv()
            print(f"<<< {response}")
            return response
        print("Websocket is not connected!")
        return None
    
    async def listen(self):
        '''
        :post: instantiated obj
        :pre: continuouslt listen for messages and call callback function
        '''
        try:
            while self.connected:
                message = await self.recieve()
                if self.callback:
                    self.callback(message)  # Send msg to another obj
        except ConnectionClosed:
            print("Connection closed")
            self.connected = False 
    
    async def connect_with_auth(self):
        '''
        :pre: existing instance
        :post: connect + authenticate to websocket
        '''
        async with self._lock:
            if not self.connected: 
                #prepare data for authentication
                key_info = {"action": "auth",
                            "key": self.key,
                            "secret": self.secret}
                #connect
                print(f"Connected to {self.uri}")
                self.connected = True
                self.websocket = await connect(self.uri)

                #authenticate shortly after
                await self.send(key_info)
                await self.recieve()
                asyncio.create_task(self.listen()) #start listening to stream

    async def subscribe_to_stock(self, stock_name:str):
        stock_action = {"action": "subscribe",
            "trades": [stock_name]}
        if self.connected and self.websocket:
            await self.send(stock_action)

        
    async def close(self):
        '''
        :pre: instantiated wsc object
        :post: close websocket
        '''
        if self.connected and self.websocket:
            await self.websocket.close()
            self.connected = False
            print("Websocket connection closed")









        
