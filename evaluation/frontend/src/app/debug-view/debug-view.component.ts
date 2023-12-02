import {HttpClient} from '@angular/common/http';
import {Component} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {LabelScene} from '../label-scene';
import {FaceDataFrame, IFaceDataFrame, IFrameDetails, IGazeDataFrame} from '../scene/content/models/face';
import {PhoneMesh} from '../scene/content/phone';
import {environment} from '../../environments/environment';

@Component({
  selector: 'app-debug-view',
  templateUrl: './debug-view.component.html',
  styleUrls: ['./debug-view.component.css']
})
export class DebugViewComponent {

  protected currentFrame = 0;

  protected scene = new LabelScene();
  protected currentThread?: any;
  protected threshold = .2;
  protected position = 0;
  protected currentLabelText = "Not Set";

  protected currentVideo?: string;
  protected currentFaceModel?: string;
  protected currentGazeModel?: string;
  protected maxFrames = 0;
  protected currentFaceImage?: string;
  protected currentLabel?: number;

  protected currentGaze?: IGazeDataFrame;
  protected currentFace?: IFaceDataFrame;

  protected static readonly PARAMS: { [key: string]: { [key: string]: { size: number, position: number } } } = {
    "dlib": {
      "eth-gaze": {
        "size": 0.204,
        "position": 0.022
      },
      "l2cs-gaze-360": {
        "size": 0.5,
        "position": 0.047
      },
      "mpiifacegaze": {
        "size": 0.082,
        "position": -0.027
      },
      "head-rotation": {
        "size": 0.265,
        "position": 0.206
      }
    },
    "mediapipe": {
      "eth-gaze": {
        "size": 0.286,
        "position": 0.01
      },
      "l2cs-gaze-360": {
        "size": 0.398,
        "position": 0.047
      },
      "mpiifacegaze": {
        "size": 0.051,
        "position": -0.027
      },
      "head-rotation": {
        "size": 0.296,
        "position": 0.206
      }
    }
  }

  constructor(private http: HttpClient, private route: ActivatedRoute) {
  }

  onInputChange(event: any) {
    console.log(event);
    const text = event.target.value;
    const numericPattern = /^\d*$/;

    if (!numericPattern.test(text)) {

      event.target.value = `${this.currentFrame}`;
      return;
    }

    const frame = parseInt(text);
    const roundedFrame = Math.round(frame / 3) * 3;
    event.target.value = `${roundedFrame}`;
    this.setFrame(roundedFrame);
  }

  protected onChange(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target === null) throw new Error("Illegal State");

    this.threshold = Number(target.value);
    PhoneMesh.HEIGHT = this.threshold;
    this.scene.getPhone().update();
  }

  protected onChangePosition(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target === null) throw new Error("Illegal State");

    this.position = Number(target.value);
    PhoneMesh.POSITION = this.position;
    this.scene.getPhone().update();
  }

  getLabel(value?: number) {
    if (value === 2) return "?";
    if (value === 1) return "Schaut"
    if (value === 0) return "Schaut nicht."
    return "Not set."
  }

  ngAfterViewInit(): void {

    this.scene.getCamera().position.x = -1.;
    this.scene.getCamera().position.y = .3
    this.scene.getCamera().position.z = .2;
    this.scene.setCameraFocus(0, 0, 0);

    this.route.params.subscribe(params => {
        this.currentVideo = params["video"];
        this.currentFaceModel = params["face"]
        this.currentGazeModel = params["gaze"]

        const frame = this.route.snapshot.queryParams['frame'];
        this.currentFrame = (frame === undefined || frame == null) ? 0 : Math.round(parseInt(frame) / 3) * 3;


        this.http.get(`${environment.apiUrl}/api/size/${this.currentVideo}/${this.currentFaceModel}/${this.currentGazeModel}`) // Replace with your API endpoint
          .subscribe((data) => {
            this.maxFrames = (data as { size: number }).size * 3;
            this.setFrame(this.currentFrame);

            this.position = DebugViewComponent.PARAMS[this.currentFaceModel!][this.currentGazeModel!].position;
            PhoneMesh.POSITION = this.position;

            this.threshold = DebugViewComponent.PARAMS[this.currentFaceModel!][this.currentGazeModel!].size;
            PhoneMesh.HEIGHT = this.threshold;

            this.scene.getPhone().update();
          });
      }
    );
  }

  protected next() {
    this.setFrame(this.currentFrame + 3);
  }

  protected previous() {
    this.setFrame(this.currentFrame - 3);
  }


  public setFrame(frameNumber: number) {
    this.currentFrame = frameNumber;
    if (this.currentFrame < 0) {
      this.currentFrame = 0;
    }


    if (this.currentVideo === undefined) {
      return;
    }

    if (this.currentFrame > this.maxFrames) {
      this.currentFrame = this.maxFrames;
    }

    this.http.get(`${environment.apiUrl}/api/frame/${this.currentVideo}/${this.currentFaceModel}/${this.currentGazeModel}/${this.currentFrame}`)
      .subscribe((data) => {
        data = data as IFrameDetails;
        this.currentLabel = (data as IFrameDetails).label!;
        this.currentLabelText = this.getLabel(this.currentLabel) +  " " + this.currentLabel + " " + this.currentFrame ;
        if ((data as IFrameDetails).face === undefined) {
          this.scene.load();
          this.currentGaze = undefined;
          this.currentFaceImage = undefined;
          this.currentFace = undefined;
          return;
        }
        this.currentFace = (data as IFrameDetails).face!;
        this.currentGaze = (data as IFrameDetails).gaze!;

        this.currentFaceImage = `${environment.apiUrl}/api/faceImage/${this.currentVideo}/${this.currentFaceModel}/${this.currentFrame}`
        const faceDataFrame = new FaceDataFrame((data as IFrameDetails).face!, (data as IFrameDetails).gaze!);
        this.scene.load(faceDataFrame);
      });
  }

  showError() {
    if (this.currentFace === undefined && this.currentLabel === 1) {
      return true;
    }
    if (this.currentFace === undefined) {
      return false;
    }

    if (this.currentLabel === 1 && !this.currentGaze!.hit) {
      return true;
    }

    if (this.currentLabel === 0 && this.currentGaze!.hit) {
      return true;
    }

    return false;
  }

  protected toggle() {
    if (this.currentThread === undefined) {
      this.animate();
      return;
    }
    clearTimeout(this.currentThread);
    this.currentThread = undefined;
  }

  protected animate() {
    this.currentThread = setTimeout(() => {
      this.setFrame(this.currentFrame + 3);
      if (this.showError()) {
        this.toggle();
        return;
      }
      this.animate();
    }, 100)
  }

  protected getIcon() {
    return this.currentThread === undefined ? "play_arrow" : "pause";
  }

  protected getImageUrl() {
    if (this.currentVideo === undefined) return "";

    return `${environment.apiUrl}/api/baseImage/${this.currentVideo}/${this.currentFrame}`;
  }

  protected radToDeg(value
                       :
                       number
  ) {
    return value * 180 / Math.PI;
  }

  protected getHeading() {
    if (this.currentVideo === undefined) return "";
    return `${this.currentVideo}/${this.currentFaceModel}/${this.currentGazeModel} - ${this.currentFrame} - ${this.maxFrames}`
  }

}
