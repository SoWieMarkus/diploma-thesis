import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LabelBlinkViewComponent } from './label-blink-view.component';

describe('LabelBlinkViewComponent', () => {
  let component: LabelBlinkViewComponent;
  let fixture: ComponentFixture<LabelBlinkViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LabelBlinkViewComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LabelBlinkViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
