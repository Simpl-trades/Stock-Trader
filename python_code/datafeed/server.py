import asyncio
from stream_observer import DataProcessor
from web_socket_client import WebSocketClient


# Run the WebSocket connection
async def main(ws_client, data_processor):
    await ws_client.connect_with_auth()
    await ws_client.subscribe_to_stock("FAKEPACA")
    await asyncio.sleep(20)  # Keep the connection open for streaming
    await ws_client.close()

if __name__ == "__main__":
    file_object = open(r"important.txt")
    info = file_object.readlines()
    key = info[0].replace("\n", "")
    secret = info[1].replace("\n", "")
    uri = "wss://stream.data.alpaca.markets/v2/test"


    # Instantiate the data processor
    data_processor = DataProcessor()

    # Instantiate WebSocketClient and pass the processor’s method as the callback
    ws_client = WebSocketClient(uri,
                                key,
                                secret,
                                callback=data_processor.process_data)

    # Run the async function
    asyncio.run(main(ws_client, data_processor))
