from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class FrameSaver:
    def __init__(self, filename, frames_dir, show=False, interval=100):
        frames_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.frames_dir = Path(frames_dir) / (filename + "_frames")
        self.show = show
        self.interval = interval
        self.frame_num = 0

        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(self, fig):
        next_frame_filename = f"frame{self.frame_num}.png"
        self.frame_num += 1
        fig.savefig(self.frames_dir / next_frame_filename)

    def create_video_from_frames(self, video_format="html"):
        frame_files = sorted(self.frames_dir.glob("frame*.png"), key=lambda x: int(x.stem[5:]))
        fig, ax = plt.subplots()
        img = plt.imread(frame_files[0])
        im = ax.imshow(img, cmap="gray")
        ax.axis("off")
        plt.tight_layout()

        def update(frame):
            frame_img = plt.imread(frame)
            im.set_array(frame_img)
            return [im]

        ani = FuncAnimation(fig, update, frames=frame_files, interval=self.interval, blit=True)
        suffix, writer = {"html": (".html", "html"), "gif": (".gif", "pillow")}.get(video_format)
        output_filename = self.filename + suffix
        ani.save(output_filename, writer=writer)

        if self.show:
            import webbrowser  # hehehe ok 1 import
            webbrowser.open(output_filename)
