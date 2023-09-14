import array
from Neural_network_processing import INeural
from typing import List
import json


def gen_message(orderId, neuralType, textResult=None, images=None, errorMsg=None):
    message = {
        "orderId": orderId,
        "neuralType": neuralType,
        "textResult": textResult,
        "images": images,
        "errorMsg": errorMsg
    }
    return message


def gen_responce(body, message, exchange):
    response = {
        "messageId": body['messageId'],
        "conversationId": body['conversationId'],
        "sourceAddress": body['sourceAddress'],
        "destinationAddress": body['destinationAddress'],
        "requestId": body['requestId'],
        "messageType": ['urn:message:' + exchange],
        "message": message
    }
    return json.dumps(response)


def run_neurals(msg):
    msg_response = {}
    if msg['neuralType'] == 'colorizer' or msg['neuralType'] == 'delete_background':
        result = INeural.colorizer(
            init_img_binary_data=msg['imagesInput'][0], params=json.loads(msg['params']))
        if result is bytes:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], images=[result])
    elif msg['neuralType'] == 'upscaler':
        if msg['caption']:
            result = INeural.upscaler(
                init_img_binary_data=msg['imagesInput'][0], caption=msg['caption'], params=json.loads(msg['params']))
        else:
            result = INeural.upscaler(
                init_img_binary_data=msg['imagesInput'][0], params=json.loads(msg['params']))
        if result is bytes:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], images=[result])
    elif msg['neuralType'] == 'image_to_image':
        result = INeural.image_to_image(
            init_img_binary_data=msg['imagesInput'][0], caption=msg['caption'], params=json.loads(msg['params']))
        if result is List[bytes]:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], images=array("i", result))
    elif msg['neuralType'] == 'text_to_image':
        result = INeural.text_to_image(
            caption=msg['caption'], params=json.loads(msg['params']))
        if result is List[bytes]:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], images=array("i", result))
    elif msg['neuralType'] == 'image_captioning':
        result = INeural.image_captioning(
            init_img_binary_data=msg['imagesInput'][0], caption=msg['caption'], params=json.loads(msg['params']))
        if result is str:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], textResult=result)
    elif msg['neuralType'] == 'image_classification':
        result = INeural.image_classification(
            init_img_binary_data=msg['imagesInput'][0])
        if result is List[int]:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], textResult=str(result))
    elif msg['neuralType'] == 'translation':
        param = json.loads(msg['params'])
        final_source_lang, translated_text = INeural.translation(
            input_text=param['input_text'], source_lang=param['source_lang'], dest_lang=param['dest_lang'])
        if result is tuple:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], textResult=translated_text)
    elif msg['neuralType'] == 'inpainting':
        result = INeural.inpainting(
            init_img_binary_data=msg['imagesInput'][0], mask_binary_data=msg['imagesInput'][1], caption=msg['caption'], params=json.loads(msg['params']))
        if result is List[bytes]:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], images=array("i", result))
    elif msg['neuralType'] == 'stylization':
        result = INeural.stylization(
            content_binary_data=msg['imagesInput'][0], style_binary_data=msg['imagesInput'][1], prompt=msg['prompts'][0], params=json.loads(msg['params']))
        if result is List[bytes]:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], images=array("i", result))
    elif msg['neuralType'] == 'image_fusion':
        result = INeural.image_fusion(img1_binary_data=msg['imagesInput'][0], img2_binary_data=msg['imagesInput']
                                      [1], prompt1=msg['prompts'][0], prompt2=msg['prompts'][1], params=json.loads(msg['params']))
        if result is List[bytes]:
            msg_response = gen_message(
                msg['orderId'], msg['neuralType'], images=array("i", result))
    if not bool(msg_response):
        msg_response = gen_message(
            msg['orderId'], msg['neuralType'], errorMsg="error")

    return msg_response
