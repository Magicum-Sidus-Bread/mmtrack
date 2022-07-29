# run mot demo
import mmcv
import tempfile
from mmtrack.apis import inference_mot, init_model
mot_config = '../configs/mot/myconfig/ori_deepsort.py'
input_video = './outout.mp4'
imgs = mmcv.VideoReader(input_video)
# build the model from a config file
mot_model = init_model(mot_config, device='cuda:0')
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
# test and show/save the images
for i, img in enumerate(imgs):
    print("img")
    print(len(img))
    result = inference_mot(mot_model, img, frame_id=i)
    mot_model.show_result(
            img,
            result,
            show=False,
            wait_time=int(1000. / imgs.fps),
            out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()

output = './output_completed3.mp4'
print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()