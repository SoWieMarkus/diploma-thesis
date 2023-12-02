import { getSubDirectories, OUTPUT_PATH } from "./helper";
import { Video } from "./video";


export class VideoManager {

    private videos: Video[] = [];

    constructor() {
        this.refresh();
    }

    public refresh() {
        const videoPaths = getSubDirectories(OUTPUT_PATH);

        for (const directory of videoPaths) {
            this.videos.push(new Video(directory))
        }
    }

    public getVideo(name: string) {
        for (const video of this.videos) {
            if (video.getName() === name) {
                return video;
            }
        }
        return undefined;
    }

    public getDetails() {
        const allVideos = [];
        for (const video of this.videos) {
            allVideos.push({
                name: video.getName(),
                details: video.getDetailsGaze()
            })
        }
        return allVideos;
    }

}
