# pp det to onnx
paddle2onnx --model_dir ./ch_PP-OCRv4_det_server_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./onnx/ppv4_det.onnx --opset_version 16 --enable_onnx_checker True

paddle2onnx --model_dir ./ch_PP-OCRv4_rec_server_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./onnx/ppv4_rec.onnx --opset_version 16 --enable_onnx_checker True

paddle2onnx --model_dir ./ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./onnx/ppv4_cls.onnx --opset_version 16 --enable_onnx_checker True
