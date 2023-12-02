import { Component } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment'

interface Set {
  redirect: string;
  size: number;
  missing: number;
  done: number;
  video: string;
  face?: string;

}

interface IResponse {
  details: {
    total: number;
    missing: number;
    done: number;
  },
  links: Set[]
}



@Component({
  selector: 'app-labels-list',
  templateUrl: './labels-list.component.html',
  styleUrls: ['./labels-list.component.css']
})
export class LabelsListComponent {

  protected labels?: IResponse;

  constructor(private title: Title, private http: HttpClient, private router: Router) {
  }

  ngAfterViewInit(): void {


    this.http.get(`${environment.apiUrl}/api/labels`) // Replace with your API endpoint
      .subscribe((data) => {
        this.labels = data as IResponse;
        this.labels.links.sort((a, b) => {
          if (a.missing === 0 && b.missing !== 0) return 1;
          if (a.missing !== 0 && b.missing === 0) return -1;

          if (a.missing < b.missing) return -1;
          if (a.missing > b.missing) return 1;

          if (a.video < b.video) return -1;
          if (a.video > b.video) return 1;

          return 0;
        });
      });
  }

  redirect(url: string) {
    this.router.navigate([url]);
  }

  getType(dataSet: Set) {
    if (dataSet.face === undefined) return "Blickrichtung"
    if (dataSet.face === "blink") return "Blinzelerkennung"
    return "Gesichtserkennung"
  }

  getHeaderColor(dataSet: Set) {
    if (dataSet.face === undefined) return "#819FF7";
    if (dataSet.face === "blink") return "#32CD32"
    return "#FE9A2E"
  }

  getOpacity(dataSet: Set) {
    if (dataSet.missing === 0) return 0.3;
    return 1;
  }

  getProgress() {
    return (this.labels!.details.done / this.labels!.details.total) * 100;
  }

}
