set PYTHON_PATH=python
set SCRIPT_PATH=eval_script.py

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "clip" --use_generated 1 --image_size 512
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "flava" --use_generated 1 --image_size 512
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "siglip" --use_generated 1 --image_size 512
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "flava" --use_generated 0 --image_size 512
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "dalle3" --model_name "clip" --use_generated 0 --image_size 512

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 32 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 256 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 32 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 256 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 32 --output_dir "Flamingo"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "Flamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "photoOfBirdFlamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaDalle"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"



@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "FinalCLIPSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 256 --output_dir "FinalCLIPSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 32 --output_dir "FinalCLIPSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "photoOfBirdFlamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "photoOfBirdFlamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 256 --output_dir "photoOfBirdFlamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 32 --output_dir "photoOfBirdFlamingo"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalCLIPSDXLPhoto"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 32 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 1024 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 256 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 32 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 1024 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 256 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 32 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXL"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 32 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 1024 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 256 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 32 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 1024 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 256 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 32 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXL"




@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "clip" --use_generated 1 --image_size 32 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "dalle3" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "dalle3" --model_name "clip" --use_generated 1 --image_size 256 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "dalle3" --model_name "clip" --use_generated 1 --image_size 32 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "dalle3" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "dalle3" --model_name "clip" --use_generated 0 --image_size 256 --output_dir "FinalClipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "dalle3" --model_name "clip" --use_generated 0 --image_size 32 --output_dir "FinalClipDalle"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "flava" --use_generated 1 --image_size 32 --output_dir "FinalFlavaDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "dalle3" --model_name "flava" --use_generated 1 --image_size 1024 --output_dir "FinalFlavaDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "dalle3" --model_name "flava" --use_generated 1 --image_size 256 --output_dir "FinalFlavaDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "dalle3" --model_name "flava" --use_generated 1 --image_size 32 --output_dir "FinalFlavaDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "dalle3" --model_name "flava" --use_generated 0 --image_size 1024 --output_dir "FinalFlavaDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "dalle3" --model_name "flava" --use_generated 0 --image_size 256 --output_dir "FinalFlavaDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "dalle3" --model_name "flava" --use_generated 0 --image_size 32 --output_dir "FinalFlavaDalle"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "siglip" --use_generated 1 --image_size 32 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "dalle3" --model_name "siglip" --use_generated 1 --image_size 1024 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "dalle3" --model_name "siglip" --use_generated 1 --image_size 256 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "dalle3" --model_name "siglip" --use_generated 1 --image_size 32 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "dalle3" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "dalle3" --model_name "siglip" --use_generated 0 --image_size 1024 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "dalle3" --model_name "siglip" --use_generated 0 --image_size 256 --output_dir "FinalSiglipDalle"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "dalle3" --model_name "siglip" --use_generated 0 --image_size 32 --output_dir "FinalSiglipDalle"


@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 1024 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 1024 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 1024 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"  --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"  --local 0

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 1024 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"


@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "FinalClipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "FinalCLIPSDXLPhoto"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 1024 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 1024 --output_dir "FinalFlavaSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 1024 --output_dir "FinalSiglipSDXL"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 1024 --output_dir "FinalSiglipSDXL"

@REM ______________________________________________________________________________________________________________

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 1024 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 1024 --output_dir "FinalSiglipSDXLCaptions"


@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 1024 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 1024 --output_dir "FinalFlavaSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "FinalClipSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 512 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "siglip" --use_generated 0 --image_size 1024 --output_dir "FinalSiglipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "siglip" --use_generated 1 --image_size 1024 --output_dir "FinalSiglipSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 512 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "flava" --use_generated 0 --image_size 1024 --output_dir "FinalFlavaSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "flava" --use_generated 1 --image_size 1024 --output_dir "FinalFlavaSDXLCaptions"

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "FinalClipSDXLCaptions"
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "FinalClipSDXLCaptions"

@REM ______________________________________________________________________________________________________________

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "imagenet" --dataset_link "clip-benchmark/wds_imagenet1k" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "extraDatasets"


@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "vegfru" --dataset_link "veg" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "newCLIP" --local 1
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "vegfru" --dataset_link "fru" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "newCLIP2" --local 1
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "vegfru" --dataset_link "veg" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "newCLIP" --local 1
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "vegfru" --dataset_link "fru" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "newCLIP2" --local 1


@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "butterflies" --dataset_link "butterflies" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "newCLIP" --local 1
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "butterflies" --dataset_link "butterflies" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "newCLIP" --local 1


@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "textures" --dataset_link "cansa/Describable-Textures-Dataset-DTD" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "newCLIP" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "textures" --dataset_link "cansa/Describable-Textures-Dataset-DTD" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "newCLIP" --local 0

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "gemstones" --dataset_link "gemstones" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 256 --output_dir "newCLIP" --local 1
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "gemstones" --dataset_link "gemstones" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 256 --output_dir "newCLIP" --local 1

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sneakers" --dataset_link "sneakers" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 140 --output_dir "gen_images1" --local 1
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sneakers" --dataset_link "sneakers" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 140 --output_dir "newCLIP" --local 1

@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 256 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 32 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cub" --dataset_link "Donghyun99/CUB-200-2011" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 512 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "aircraft" --dataset_link "clip-benchmark/wds_fgvc_aircraft" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 1024 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "resisc" --dataset_link "clip-benchmark/wds_vtab-resisc45" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 256 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cifar" --dataset_link "clip-benchmark/wds_vtab-cifar100" --generation_tool "sdxl" --model_name "clip" --use_generated 0 --image_size 32 --output_dir "top3" --local 0


@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "pets" --dataset_link "clip-benchmark/wds_vtab-pets" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "flowers" --dataset_link "clip-benchmark/wds_vtab-flowers" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "food" --dataset_link "clip-benchmark/wds_food101" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "cars" --dataset_link "clip-benchmark/wds_cars" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "sun" --dataset_link "clip-benchmark/wds_sun397" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 512 --output_dir "top3" --local 0
@REM %PYTHON_PATH% %SCRIPT_PATH% --dataset_name "objectnet" --dataset_link "clip-benchmark/wds_objectnet" --generation_tool "sdxl" --model_name "clip" --use_generated 1 --image_size 1024 --output_dir "top3" --local 0
