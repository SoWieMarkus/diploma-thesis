import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { MatSidenavModule } from "@angular/material/sidenav";
import { MatIconModule } from "@angular/material/icon";

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { SidebarComponent } from './sidebar/sidebar.component';
import { DebugViewComponent } from './debug-view/debug-view.component';
import { SceneComponent } from './scene/scene.component';
import { HttpClientModule } from '@angular/common/http';
import { MatButtonModule } from "@angular/material/button";
import { MatProgressBarModule } from "@angular/material/progress-bar";
import { MatExpansionModule } from "@angular/material/expansion"
import { MatInputModule } from "@angular/material/input"
import { MatSliderModule } from "@angular/material/slider";
import { LandingViewComponent } from './landing-view/landing-view.component'
import { MatListModule } from "@angular/material/list"
import { MatCardModule } from "@angular/material/card";
import { LabelFaceViewComponent } from './label-face-view/label-face-view.component';
import { LabelBaseViewComponent } from './label-base-view/label-base-view.component';
import { ThankYouComponent } from './thank-you/thank-you.component';
import { LabelsListComponent } from './labels-list/labels-list.component';
import { LabelBlinkViewComponent } from './label-blink-view/label-blink-view.component'

@NgModule({
  declarations: [
    AppComponent,
    SidebarComponent,
    DebugViewComponent,
    SceneComponent,
    LandingViewComponent,
    LabelFaceViewComponent,
    LabelBaseViewComponent,
    ThankYouComponent,
    LabelsListComponent,
    LabelBlinkViewComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    MatSidenavModule,
    MatIconModule,
    HttpClientModule,
    MatButtonModule,
    MatExpansionModule,
    MatInputModule,
    MatSliderModule,
    MatListModule,
    MatCardModule,
    MatProgressBarModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
