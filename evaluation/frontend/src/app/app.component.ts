import { Component, ViewChild } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { Router } from '@angular/router';
import { SidebarComponent } from './sidebar/sidebar.component';
import { SidebarService } from './sidebar/sidebar.service';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  @ViewChild(SidebarComponent) child?: SidebarComponent;

  protected currentDataset = "test_mp_eth";

  private isShowing = true;

  protected get showSidebar() {
    return this.router.url.includes("debug")
  };

  constructor(protected sidebarService: SidebarService, private title: Title, private router: Router) { }

  public toggleSideBar() {
    this.sidebarService.toggleOpen();
  }

  protected ngOnInit() {
    this.sidebarService.isOpen.subscribe(isOpen => {
      this.isShowing = isOpen;
    });
    this.title.setTitle("Evalution: Gaze")
  }

  public getShowing() {
    return this.isShowing;
  }
}
