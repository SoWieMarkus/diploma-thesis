import cors from "cors";
import express from "express";
import path from "path";
import { router } from "./src/router/routing";

const app = express();
const port = 1320;

const staticDir = path.join(__dirname, '../../frontend/dist/');

app.use(express.json());
app.use(cors());
app.use(express.static(staticDir));

app.get("/api/baseImage/:video/:frame", (request, res) => {
    const videoName = request.params["video"];
    const frameNumber = parseInt(request.params["frame"]);

    res.sendFile(path.join(staticDir, "..", "..", "..", `output/${videoName}/images/${frameNumber}.jpg`))
})

app.get("/api/faceImage/:video/:face/:frame", (request, res) => {
    const videoName = request.params["video"];
    const faceModel = request.params["face"];
    const frameNumber = parseInt(request.params["frame"]);

    const filePath = path.join(staticDir, "..", "..", "..", `output/${videoName}/face/${faceModel}/images/${frameNumber}.jpg`)
    res.sendFile(filePath)
})



app.use("/api", router);

// Handle all routes by serving the Angular app's main HTML file
app.get('*', (req, res) => {
    res.sendFile(path.join(staticDir, 'index.html'));
});


app.listen(port, () => {
    console.log(`Server started on port ${port}`);
});
