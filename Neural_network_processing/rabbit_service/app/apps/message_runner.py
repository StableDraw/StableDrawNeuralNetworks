import array
# from Neural_network_processing import INeural

from typing import List
import json
import base64
import sys 
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(parent_dir)
# from apps import INeuralProxy
# from ....INeural import INeural as INeuralProxy
import INeural as INeuralProxy


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

def base64_to_bytes(input_img):
    img_data = input_img.encode()
    return base64.b64decode(img_data)

def bytes_to_base64(imput_bytes):
    return base64.b64encode(imput_bytes).decode('utf8')

def run_neurals(msg):
    msg_response = {}
    neural_type = msg['neuralType']
    try:
        if neural_type == 'colorizer':            
            params = json.loads(msg['parameters'])
            result = INeuralProxy.colorizer(
                init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]), params=params)                                                
            msg_response = gen_message(
                msg['orderId'], neural_type, images=[bytes_to_base64(result)])
        elif neural_type == 'delete_background':
            params = json.loads(msg['parameters'])
            result = INeuralProxy.delete_background( init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]), params=params)
            msg_response = gen_message(
                msg['orderId'], neural_type, images=[bytes_to_base64(result)])
        elif neural_type == 'upscaler':
            if msg['caption']:
                result = INeuralProxy.upscaler(
                    init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]), caption=msg['caption'], params=json.loads(msg['parameters']))
            else:
                result = INeuralProxy.upscaler(
                    init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]), params=json.loads(msg['parameters']))        
            msg_response = gen_message(
                    msg['orderId'], neural_type, images=[bytes_to_base64(result)])
        elif neural_type == 'image_to_image':
            result = INeuralProxy.image_to_image(
                init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]), caption=msg['caption'], params=json.loads(msg['parameters']))
            if result is List[bytes]:
                msg_response = gen_message(
                    msg['orderId'], neural_type, images=array("i", result))
        elif neural_type == 'text_to_image':
            result = INeuralProxy.text_to_image(
                caption=msg['caption'], params=json.loads(msg['parameters']))
            if result is List[bytes]:
                msg_response = gen_message(
                    msg['orderId'], neural_type, images=array("i", result))
        elif neural_type == 'image_captioning':
            print(json.loads(msg['parameters']))
            result = INeuralProxy.image_captioning(
                init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]), caption=msg['caption'], params=json.loads(msg['parameters']))
            if result is str:
                msg_response = gen_message(
                    msg['orderId'], neural_type, textResult=result)
        elif neural_type == 'image_classification':
            result = INeuralProxy.image_classification(
                init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]))
            if result is List[int]:
                msg_response = gen_message(
                    msg['orderId'], neural_type, textResult=str(result))
        elif neural_type == 'translation':
            param = json.loads(msg['parameters'])
            final_source_lang, translated_text = INeuralProxy.translation(
                input_text=param['input_text'], source_lang=param['source_lang'], dest_lang=param['dest_lang'])
            if result is tuple:
                msg_response = gen_message(
                    msg['orderId'], neural_type, textResult=translated_text)
        elif neural_type == 'inpainting':
            result = INeuralProxy.inpainting(
                init_img_binary_data=base64_to_bytes(msg['imagesInput'][0]), mask_binary_data=msg['imagesInput'][1], caption=msg['caption'], params=json.loads(msg['parameters']))
            if result is List[bytes]:
                msg_response = gen_message(
                    msg['orderId'], neural_type, images=array("i", result))
        elif neural_type == 'stylization':
            result = INeuralProxy.stylization(
                content_binary_data=base64_to_bytes(msg['imagesInput'][0]), style_binary_data=msg['imagesInput'][1], prompt=msg['prompts'][0], params=json.loads(msg['parameters']))
            if result is List[bytes]:
                msg_response = gen_message(
                    msg['orderId'], neural_type, images=array("i", result))
        elif neural_type == 'image_fusion':
            result = INeuralProxy.image_fusion(img1_binary_data=base64_to_bytes(msg['imagesInput'][0]), img2_binary_data=msg['imagesInput']
                               [1], prompt1=msg['prompts'][0], prompt2=msg['prompts'][1], params=json.loads(msg['parameters']))
            if result is List[bytes]:
                msg_response = gen_message(
                    msg['orderId'], neural_type, images=array("i", result))
        if not bool(msg_response):
            msg_response = gen_message(
                msg['orderId'], neural_type, errorMsg="error")
    except Exception as ex:
        msg_response = gen_message(
                msg['orderId'], neural_type, errorMsg=ex.__str__())
    finally:
        return msg_response
