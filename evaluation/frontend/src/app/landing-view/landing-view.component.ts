import { Component } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment'
export interface IOverviewVideo {
  name: string;
  details: { face: string, gaze: string, frames: number }[];
}

@Component({
  selector: 'app-landing-view',
  templateUrl: './landing-view.component.html',
  styleUrls: ['./landing-view.component.css']
})
export class LandingViewComponent {

  protected dataSets?: IOverviewVideo[];

  constructor(private title: Title, private http: HttpClient) { }

  ngAfterViewInit(): void {


    this.http.get(`${environment.apiUrl}/api/overview`) // Replace with your API endpoint
      .subscribe((data) => {
        this.dataSets = data as IOverviewVideo[];
      });
  }

  getLink(name: string, details: { face: string, gaze: string, frames: number }) {
    return `debug/${name}/${details.face}/${details.gaze}`
  }

  getThumbnail(name: string) {
    return `${environment.apiUrl}/api/baseImage/${name}/0`
  }

}
