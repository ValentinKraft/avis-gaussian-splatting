python train.py --model_path _output_/densify-test --mask_path _input_/minimask-binary3.nii.gz --volume_path _input_/minict.nii.gz --iterations 2000 --init_n_points 10000 --medical_mode none --enable_densification --densify_from_iter 40 --densify_until_iter 1850 --densification_interval 15 --densify_grad_threshold 0.0 --prune_min_opacity 0.003 --opacity_reset_interval 0 --max_points_per_iter 20000 --volume_downscale_factor 1 --volume_render_downscale_factor 1 --disable_volume_overflow_guard --volume_storage_dtype fp16 --save_ply_every 100 --enable_diagnostics 

----

wsl -d Ubuntu-22.04
conda activate nerfstudio

ns-process-data images --data C:\DEV\TESTS\gs\_SCENES_\_scene_ --output-dir C:\DEV\TESTS\gs\_SCENES_\NERFSTUDIO\AHrEZ-800 --colmap-cmd C:\DEV\TESTS\gs\COLMAP\bin\colmap.exe --matching-method exhaustive --camera-type simple_pinhole --num-downscales 0

ns-train splatfacto --data /mnt/c/DEV/TESTS/gs/_SCENES_/NERFSTUDIO/AHrEZ-200-png --mixed-precision True --pipeline.model.sh-degree 2

ns-train splatfacto --data /mnt/c/DEV/TESTS/gs/_SCENES_/NERFSTUDIO/a
bdomen --pipeline.model.sh-degree 2 --pipeline.model.stop-split-at 50000 --max-num-iterations 6000 --pipeline.model.dens
ify-grad-thresh 0.0001

ns-export gaussian-splat --load-config outputs/MINICT/splatfacto/2026-02-28_163704/config.yml --output-dir /mnt/c/DEV/TESTS/gs/_SCENES_

ns-viewer --load-config ...

conda activate cin3dgs
C:\DEV\TESTS\gs\COLMAP\bin\colmap.exe model_converter --input_path C:\DEV\TESTS\gs\avis-gaussian-splatting\_scene_\sparse\0 --output_path C:\DEV\TESTS\gs\avis-gaussian-splatting\_scene_\sparse\0 --output_type BIN
C:\DEV\TESTS\gs\viewers\bin\SIBR_gaussianViewer_app.exe -m C:\DEV\TESTS\gs\avis-gaussian-splatting\output\...
python train.py -s scene -m model --eval --test_iterations 7000 15000 30000 --densify_grad_threshold 0.00005 --save_iterations 30000
python train.py -s C:\DEV\TESTS\gs\_scene_ --eval --test_iterations 7000 15000 30000 --save_iterations 10000 -r 2
python train.py -s _scene_ --save_iterations 1 -r 2
python train.py -s C:\DEV\TESTS\gs\_scene_-RANDOM -m model-random
