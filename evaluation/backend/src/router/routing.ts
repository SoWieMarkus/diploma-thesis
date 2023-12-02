import express from "express";
import { VideoManager } from "../io/video-manager";
import { LabelManager } from "../label";

export const router = express.Router();

// TODO make singleton
const videoManager = new VideoManager();
const labelManger = LabelManager.getInstance();

router.get("/overview", (request, response) => {
    response.json(videoManager.getDetails());
})

router.get("/size/:video/:face/:gaze", (request, response) => {
    const video = videoManager.getVideo(request.params["video"]);

    if (video === undefined) {
        response.status(404).json();
        return;
    }

    response.json({ size: video.getSize(request.params["face"], request.params["gaze"]) })
})

router.get("/frame/:video/:face/:gaze/:frame", (request, response) => {

    const name = request.params["video"];
    const face = request.params["face"];
    const gaze = request.params["gaze"];
    const frameNumber = parseInt(request.params["frame"]);

    const video = videoManager.getVideo(name);
    if (video === undefined) {
        response.status(404).json("Video not found.");
        return;
    }

    const details = video.getByFrameNumber(face, gaze, frameNumber);
    if (details === undefined) {
        response.status(404).send("Details not found.");
        return;
    }

    const labelSet = labelManger.getLabelset(name);
    const label = labelSet === undefined ? undefined : labelSet.getByFrameNumber(frameNumber);
    details.label = label;
    response.json(details);
});

router.get("/labelinit/base/:video", (request, response) => {
    const name = request.params["video"];
    const labelSet = labelManger.getLabelset(name);

    if (labelSet === undefined) {
        response.status(404).json({
            message: "No labelset found"
        });
        return;
    }

    response.json({
        first: labelSet.getFirstMissing(),
        size: labelSet.getSize()
    })
})

router.get("/labels", (request, response) => {
    response.json(labelManger.getDetails())
})

router.get("/labelinit/face/:video/:face", (request, response) => {

    const name = request.params["video"];
    const face = request.params["face"];
    const labelSet = labelManger.getLabelset(name, face);

    if (labelSet === undefined) {
        response.status(404).json({
            message: "No labelset found"
        });
        return;
    }

    response.json({
        first: labelSet.getFirstMissing(),
        size: labelSet.getSize()
    })

})

router.get("/labelprev/base/:video/:frame", (request, response) => {

    const name = request.params["video"];
    const frame = request.params["frame"]
    const labelSet = labelManger.getLabelset(name);

    if (labelSet === undefined) {
        response.status(404).json({
            message: "No labelset found"
        });
        return;
    }

    response.json(labelSet.previous(parseInt(frame)))

})


router.get("/labelprev/face/:video/:face/:frame", (request, response) => {

    const name = request.params["video"];
    const frame = request.params["frame"]
    const face = request.params["face"]
    const labelSet = labelManger.getLabelset(name, face);

    if (labelSet === undefined) {
        response.status(404).json({
            message: "No labelset found"
        });
        return;
    }
    response.json(labelSet.previous(parseInt(frame)))
})

router.post("/label", (request, response) => {

    const body = request.body;

    const videoName = body.video;
    const frameNumber = body.frame;
    const value = body.value;

    if (videoName === undefined) {
        response.status(404).json({
            message: `No video provided in request.`
        })
        return;
    }

    if (frameNumber === undefined) {
        response.status(404).json({
            message: `No frameNumber provided in request.`
        })
        return;
    }

    if (value === undefined) {
        response.status(404).json({
            message: `No frameNumber provided in request.`
        })
        return;
    }

    const faceModelName = body.face;

    const labelSet = labelManger.getLabelset(videoName, faceModelName);

    if (labelSet === undefined) {
        response.status(404).json({
            message: `No label set found for ${videoName}/${faceModelName}.`
        })
        return;
    }

    labelSet.set(frameNumber, value);
    labelSet.save();

    response.json(labelSet.next(frameNumber));
})
