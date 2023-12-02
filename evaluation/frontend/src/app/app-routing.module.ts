import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DebugViewComponent } from './debug-view/debug-view.component';
import { LandingViewComponent } from './landing-view/landing-view.component';
import { LabelFaceViewComponent } from './label-face-view/label-face-view.component';
import { LabelBaseViewComponent } from './label-base-view/label-base-view.component';
import { ThankYouComponent } from './thank-you/thank-you.component';
import { LabelsListComponent } from './labels-list/labels-list.component';
import { LabelBlinkViewComponent } from './label-blink-view/label-blink-view.component';

const routes: Routes = [
  { path: 'debug/:video/:face/:gaze', component: DebugViewComponent },
  { path: '', component: LandingViewComponent },
  { path: 'label/face/:video/:face', component: LabelFaceViewComponent },
  { path: 'label/base/:video', component: LabelBaseViewComponent },
  { path: 'label/blink/:video', component: LabelBlinkViewComponent },
  { path: 'thanks', component: ThankYouComponent },
  { path: 'labels', component: LabelsListComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
