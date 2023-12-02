import { Component } from '@angular/core';

@Component({
  selector: 'app-thank-you',
  templateUrl: './thank-you.component.html',
  styleUrls: ['./thank-you.component.css']
})
export class ThankYouComponent {

  ngAfterViewInit(): void {

    const audio = new Audio();
    audio.src =
      Math.random() < 0.5
        ? "assets/thanks.mp3"
        : "assets/thanks_alex.mp3"

    console.log(audio.src)
    audio.volume = 0.6
    audio.load();
    audio.play();
  }

}
