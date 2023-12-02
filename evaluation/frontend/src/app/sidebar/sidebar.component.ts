import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { IOverviewVideo } from '../landing-view/landing-view.component';
import { SidebarService } from './sidebar.service';
import { environment } from '../../environments/environment'



@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})
export class SidebarComponent {

  constructor(private sidebarService: SidebarService, private http: HttpClient) { }

  protected dataSets: IOverviewVideo[] = [];

  ngOnInit() {
    this.sidebarService.isOpen.subscribe(isOpen => {
      this.isShowing = isOpen;
    });
    this.http.get(`${environment.apiUrl}/api/overview`) // Replace with your API endpoint
      .subscribe((data) => {
        this.dataSets = data as IOverviewVideo[];
      });
  }

  getLink(name: string, details: { face: string, gaze: string, frames: number }) {
    return `debug/${name}/${details.face}/${details.gaze}`
  }




  private isShowing: boolean = true;

  public getShowing() {
    return this.isShowing;
  }

  public toggleVisibility() {
    this.isShowing = !this.isShowing;
  }
}
