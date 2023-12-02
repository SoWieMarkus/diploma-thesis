# Dokumentation Blickrichtungsbestimmung

Für die Blickrichtungsbestimmung müssen zunächst die benötigten Python Packages installiert werden:

```
pip install -r requirements.txt
```

Anschließend müssen alle Videos die verarbeitet werden sollen in den `./input` Ordner abgelegt werden. Dabei ist zu
beachten, dass in der aktuellen Implementierung nur Videos im MP4-Format unterstützt werden. Es lassen sich jedoch
einfach weitere Videoformate hinzufügen, indem in der `VideoManager` Klasse das entsprechende Format zu der
Liste `video_extensions` hinzugefügt wird.

Anschließend kann die Blickrichtungsbestimmung mit dem `main.py` Skript ausgeführt werden.
Dabei werden zu Beginn alle benötigten Modelle heruntergeladen. 


ACHTUNG WICHTIG ACHTUNG HIER! Sollte die Verarbeitung mit l2cs-gaze-360 fehlschlagen,
wurde der Download der Datei nicht richtig durchgeführt. In diesem Fall muss die Datei von L2CS-net manuell unter
folgender URL heruntergeladen werden:
https://drive.google.com/uc?id=18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ&export=download
Quelle: https://github.com/Ahmednull/L2CS-Net#demo

Die Datei muss dann unter `./models/l2cs-gaze-360/L2CSNet_gaze360.pkl` gespeichert werden.

Dabei wird als erstes die Gesichtserkennung in `face_detection.py` ausgeführt. Die Ergebnisse werden im `./output`
Ordner gespeichert. (`.face` Dateien)
Dann wird die Blickrichtungsbestimmung aufbauend auf den Ergebnissen der Gesichtserkennung durchgeführt. Auch die
Ergebnisse von diesem Schritt werden in dem `./output` Ordner gespeichert. (`.gaze` Dateien). Der Aufbau der Dateien ist
in meiner Diplomarbeit erklärt.

In diesen Dateien ist dann die Vorhersage ob die Person sieht oder nicht, basierend auf der in meiner Diplomarbeit modellierten Zone aufgeführt.

Wenn du für deine Videos eine Evaluation wie in meiner Diplomarbeit durchführen möchtest, muss zunächst eine Ground Truth erstellt werden.
Dafür kann das Evaluationstool verwendet werden. Siehe die README in `./evaluation`.
Dann gehe auf `localhost:4200/labels`
Dort siehst du eine Übersicht über deine Videos und die Ground Truths die erstellt werden müssen.
Arbeite alle ab.

Im Anschluss kannst du die Evaluation mit dem `evaluate.py` Skript durchführen.
