import grpc
from concurrent import futures
import stable_draw_grpc_pb2
import stable_draw_grpc_pb2_grpc
from google.protobuf.json_format import MessageToDict

from Image_caption_generator import Gen_caption as Gen_caption_gRPC
from Delete_background import Delete_background as Delete_background_gRPC
from Upscaler import Upscale as Upscale_gRPC
from Stable_diffusion import Stable_diffusion_image_to_image as Stable_diffusion_image_to_image_gRPC
from Stable_diffusion import Stable_diffusion_text_to_image as Stable_diffusion_text_to_image_gRPC
from Stable_diffusion import Stable_diffusion_depth_to_image as Stable_diffusion_depth_to_image_gRPC
from Stable_diffusion import Stable_diffusion_inpainting as Stable_diffusion_inpainting_gRPC
from Stable_diffusion import Stable_diffusion_upscaler as Stable_diffusion_upscaler_gRPC
from Stable_diffusion import Stable_diffusion_upscaler_xX as Stable_diffusion_upscaler_xX_gRPC
from Image_classifier import Get_image_class as Get_image_class_gRPC
from Image_сolorization import Image_сolorizer as Image_сolorizer_gRPC

class RouteGuideServicer(stable_draw_grpc_pb2_grpc.StableDrawGRPCServicer):
    def GenCaption(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["beam"] = int(params["beam"])
        params["max_len_a"] = int(params["max_len_a"])
        params["max_len_b"] = int(params["max_len_b"])
        params["min_len"] = int(params["min_len"])
        params["lenpen"] = int(params["lenpen"])
        params["unkpen"] = int(params["unkpen"])
        params["no_repeat_ngram_size"] = int(params["no_repeat_ngram_size"])
        params["sampling_topk"] = int(params["sampling_topk"])
        params["seed"] = int(params["seed"])
        caption = Gen_caption_gRPC(request.init_img_binary_data, params)
        return stable_draw_grpc_pb2.TextReply(caption = caption)

    def StableDiffusionTextToImage(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["steps"] = int(params["steps"])
        params["f"] = int(params["f"])
        params["ckpt"] = int(params["ckpt"])
        params["seed"] = int(params["seed"])
        w, h, binary_data = Stable_diffusion_text_to_image_gRPC(request.caption, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def StableDiffusionImageToImage(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["ddim_steps"] = int(params["ddim_steps"])
        params["f"] = int(params["f"])
        params["ckpt"] = int(params["ckpt"])
        params["seed"] = int(params["seed"])
        params["max_dim"] = int(params["max_dim"])
        w, h, binary_data = Stable_diffusion_image_to_image_gRPC(request.init_img_binary_data, request.caption, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def StableDiffusionDepthToImage(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["ddim_steps"] = int(params["ddim_steps"])
        params["ckpt"] = int(params["ckpt"])
        params["seed"] = int(params["seed"])
        params["max_dim"] = int(params["max_dim"])
        w, h, binary_data = Stable_diffusion_depth_to_image_gRPC(request.init_img_binary_data, request.caption, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def StableDiffusionInpainting(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["ddim_steps"] = int(params["ddim_steps"])
        params["ckpt"] = int(params["ckpt"])
        params["seed"] = int(params["seed"])
        params["max_dim"] = int(params["max_dim"])
        w, h, binary_data = Stable_diffusion_inpainting_gRPC(request.init_img_binary_data, request.mask_binary_data, request.caption, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def StableDiffusionUpscaler(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["ddim_steps"] = int(params["ddim_steps"])
        params["ckpt"] = int(params["ckpt"])
        params["seed"] = int(params["seed"])
        params["outscale"] = int(params["outscale"])
        params["noise_augmentation"] = int(params["noise_augmentation"])
        params["max_dim"] = int(params["max_dim"])
        w, h, binary_data = Stable_diffusion_upscaler_gRPC(request.init_img_binary_data, request.caption, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def StableDiffusionUpscalerXX(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["ddim_steps"] = int(params["ddim_steps"])
        params["ckpt"] = int(params["ckpt"])
        params["seed"] = int(params["seed"])
        params["outscale"] = int(params["outscale"])
        params["max_dim"] = int(params["max_dim"])
        w, h, binary_data = Stable_diffusion_upscaler_xX_gRPC(request.init_img_binary_data, request.caption, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def DeleteBackground(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["ckpt"] = int(params["ckpt"])
        params["seed"] = int(params["seed"])
        params["RescaleT"] = int(params["RescaleT"])
        params["cache_size"] = [int (x) for x in params["cache_size"]]
        params["input_size"] = [int (x) for x in params["input_size"]]
        params["crop_size"] = [int (x) for x in params["crop_size"]]
        w, h, binary_data = Delete_background_gRPC(request.init_img_binary_data, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def Upscale(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["model"] = int(params["model"])
        params["outscale"] = int(params["outscale"])
        params["tile"] = int(params["tile"])
        params["tile_pad"] = int(params["tile_pad"])
        params["pre_pad"] = int(params["pre_pad"])
        w, h, binary_data = Upscale_gRPC(request.init_img_binary_data, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

    def GetImageClass(self, request, _):
        c = Get_image_class_gRPC(request.init_img_binary_data)
        return stable_draw_grpc_pb2.IntReply(c = c)

    def ImageColorizer(self, request, _):
        message = MessageToDict(request)
        params = message["params"]
        params["ckpt"] = int(params["ckpt"])
        params["steps"] = int(params["steps"])
        params["render_factor"] = int(params["render_factor"])
        params["line_color_limit"] = int(params["line_color_limit"])
        w, h, binary_data = Image_сolorizer_gRPC(request.init_img_binary_data, params)
        return stable_draw_grpc_pb2.ImageReply(w = w, h = h, binary_data = binary_data)

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10))
  stable_draw_grpc_pb2_grpc.add_StableDrawGRPCServicer_to_server(RouteGuideServicer(), server)
  server.add_insecure_port('109.111.179.197:8081')
  server.start()
  print("Server started, listening on: 109.111.179.197:8081")
  server.wait_for_termination()

if __name__ == '__main__':
    serve()