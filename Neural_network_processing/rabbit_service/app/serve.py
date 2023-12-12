import array
import json
import logging
from typing import Optional
from propan.annotations import Logger

import asyncio
import anyio
from propan.brokers.rabbit import RabbitExchange, ExchangeType
from propan import PropanApp, RabbitBroker, RabbitRouter
from propan.annotations import RabbitBroker as Broker, ContextRepo
from apps.message_runner import gen_message, gen_responce, run_neurals

from config import init_settings

import os
 
os.chdir('C:/StableDraw/StableDrawNeuralNetworks/Neural_network_processing/')
print(os.getcwd())

router = RabbitRouter()
broker = RabbitBroker()
broker.include_router(router)

app = PropanApp(broker)


exchInput = RabbitExchange(
    "StableDraw.Contracts.NeuralContracts.Requests:INeuralRequest", durable=True, type=ExchangeType.FANOUT)

exchOutput = RabbitExchange(
    "StableDraw.Contracts.NeuralContracts.Replies:INeuralReply", durable=True, type=ExchangeType.FANOUT)


@broker.handle(queue="generate-neural", exchange=exchInput, retry=False)
async def base_handler(body, logger: Logger):
    try:
        body_json = json.loads(body.decode('utf8'))    
        msg = body_json['message']
        print(json.loads(msg['parameters']))
        print(str(msg['caption']))
        response = await run_neurals(msg)
        # print(response['errorMsg'])    
        response = gen_responce(
            body_json, response, "StableDraw.Contracts.NeuralContracts.Replies:INeuralReply")  
        await broker.publish(response.encode('utf-8'), queue="neural-state", exchange=exchOutput)
    except Exception as ex:
        res_msg = gen_message(msg['orderId'], msg['neural_type'], errorMsg=ex.__str__)
        res_response = gen_responce(body_json, res_msg, "StableDraw.Contracts.NeuralContracts.Replies:INeuralReply")
        await broker.publish(res_response.encode('utf-8'), queue="neural-state", exchange=exchOutput)
        
    


    
    

@app.on_startup
async def init_app(broker: Broker, context: ContextRepo, env: Optional[str] = None):
    settings = init_settings(env)
    context.set_global("settings", settings)

    logger_level = logging.DEBUG if settings.debug else logging.INFO
    app.logger.setLevel(logger_level)
    broker.logger.setLevel(logger_level)

    await broker.connect(settings.broker.url)


if __name__ == "__main__":
    anyio.run(app.run)
