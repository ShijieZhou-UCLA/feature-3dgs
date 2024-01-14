import cv2
import os
import argparse


parser = argparse.ArgumentParser(
    description=(
        "Create Lseg videos: rgb, fmap, seg, edit."
    )
)

parser.add_argument(
    "--data",
    type=str,
    help="Path including input images (and features)",
)

parser.add_argument(
    "--iteration",
    type=int,
    required=True,
    help="Chosen number of iterations"
)

parser.add_argument(
    "--fps",
    default=10,
    type=int,
    help="Chosen number of iterations"
)

parser.add_argument("--foundation_model", "-f", required=True, type=str)


def create_video_from_images(image_folder, output_video_file, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    # Determine the width and height from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()



def main(args: argparse.Namespace) -> None:
    video_folder = os.path.join(args.data, "videos_{}".format(args.iteration))
    os.makedirs(video_folder, exist_ok=True)

    if args.foundation_model == 'lseg':
        # rendered results
        for res_dir in os.listdir(os.path.join(args.data, "novel_views")):
            if "ours_{}".format(args.iteration) in res_dir:
                if 'deletion' not in res_dir and 'extraction' not in res_dir and 'color_func' not in res_dir:
                    fmap_folder = os.path.join(args.data, "novel_views", res_dir, 'feature_map')
                    fmap_video_file = os.path.join(video_folder, '{}_fmap.mp4'.format(res_dir))
                    create_video_from_images(fmap_folder, fmap_video_file, args.fps) 
                image_folder = os.path.join(args.data, "novel_views", res_dir, 'renders')
                fmap_video_file = os.path.join(video_folder, '{}.mp4'.format(res_dir))
                create_video_from_images(image_folder, fmap_video_file, args.fps) 
        # seg results
        for sub_dir in os.listdir(args.data):
            if "seg_{}".format(args.iteration) in sub_dir:
                seg_folder = os.path.join(args.data, sub_dir, "novel_views")
                seg_video_file = os.path.join(video_folder, '{}.mp4'.format(sub_dir))
                create_video_from_images(seg_folder, seg_video_file, args.fps) 

    elif args.foundation_model == 'sam':
        # rendered results
        for res_dir in os.listdir(os.path.join(args.data, "novel_views")):
            if "ours_{}".format(args.iteration) in res_dir:
                fmap_folder = os.path.join(args.data, "novel_views", res_dir, 'feature_map')
                fmap_video_file = os.path.join(video_folder, '{}_fmap.mp4'.format(res_dir))
                create_video_from_images(fmap_folder, fmap_video_file, args.fps) 
                image_folder = os.path.join(args.data, "novel_views", res_dir, 'renders')
                fmap_video_file = os.path.join(video_folder, '{}.mp4'.format(res_dir))
                create_video_from_images(image_folder, fmap_video_file, args.fps)
        # seg results
        for sub_dir in os.listdir(args.data):
            if "seg_" in sub_dir:
                for seg_dir in os.listdir(os.path.join(args.data, sub_dir)):
                    seg_folder = os.path.join(args.data, sub_dir, seg_dir)
                    seg_video_file = os.path.join(video_folder, '{}_{}.mp4'.format(sub_dir, seg_dir.split("_")[-1]))
                    create_video_from_images(seg_folder, seg_video_file, args.fps)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)