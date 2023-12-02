import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import { environment } from '../../environments/environment'

interface IResponse {
  frame: number;
  progress: number;
}

@Component({
  selector: 'app-label-base-view',
  templateUrl: './label-base-view.component.html',
  styleUrls: ['./label-base-view.component.css']
})
export class LabelBaseViewComponent {

  protected currentVideo?: string;
  protected currentFrame?: number;
  protected currentProgress = 0;
  protected size = 0;

  constructor(private http: HttpClient, private route: ActivatedRoute, private router: Router) {

  }

  protected getHeading() {
    return `${this.currentProgress} / ${this.size}`
  }

  ngAfterViewInit(): void {
    this.load();
  }

  getImage() {
    return `${environment.apiUrl}/api/baseImage/${this.currentVideo}/${this.currentFrame}`
  }


  load() {
    this.route.params.subscribe(params => {
      this.currentVideo = params["video"];
      this.http.get(`${environment.apiUrl}/api/labelinit/base/${this.currentVideo}`) // Replace with your API endpoint
        .subscribe((data) => {
          this.currentFrame = (data as { first: IResponse, size: number }).first.frame;
          this.currentProgress = (data as { first: IResponse, size: number }).first.progress;
          this.size = (data as { first: IResponse, size: number }).size;
        });

    });
  }

  post(value: number) {
    if (this.currentFrame === undefined) return;

    const temp = this.currentFrame;
    this.currentFrame = undefined;

    this.http.post(`${environment.apiUrl}/api/label`, {
      video: this.currentVideo,
      face: undefined,
      value: value,
      frame: temp
    }).subscribe(
      (response) => {

        if (response === null) {
          this.router.navigate(["/thanks"])
          return;
        }
        this.currentFrame = (response as IResponse).frame;
        this.currentProgress = (response as IResponse).progress;

      }, (error) => {
        alert("FUCK MARKUS FUCKED UP!!!!")
        console.log(error);
      });
  }

  previous() {
    if (this.currentFrame === undefined) return;
    const temp = this.currentFrame;
    this.currentFrame = undefined;
    this.http.get<{ frame: number }>(`${environment.apiUrl}/api/labelprev/base/${this.currentVideo}/${temp}`)
      .subscribe((data) => {
        this.currentFrame = (data as IResponse).frame;
        this.currentProgress = (data as IResponse).progress;
      });

  }

}
