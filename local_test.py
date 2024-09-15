import asyncio
from lambda_function import lambda_handler

async def test_lambda():
    event = {}
    context = {}
    result = await lambda_handler(event, context)
    print(result)

if __name__ == "__main__":
    asyncio.run(test_lambda())