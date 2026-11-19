# Rechteckiger MSMA-Prüfkörper: FEM-Prototyp

Dieses Paket enthält einen eigenständigen Python-FEM-Code für einen rechteckigen
MSMA-Körper in einer rechteckigen Luftbox. `msma_material.py` ist die zuvor
erstellte Materialroutine; sie liegt unverändert bei. Die Erweiterungen für die
FEM und eine ausdrücklich zuschaltbare zusätzliche Dissipation stehen in
`msma_rectangle_fem.py`. Es sind keine kommerziellen FEM-Bibliotheken erforderlich.

## Start

Alle Dateien gemeinsam entpacken. Im Terminal derselben Python-Umgebung, die
Spyder verwendet, einmal installieren:

```text
python -m pip install -r requirements.txt
```

Danach `msma_rectangle_fem.py` in Spyder ausführen oder im Terminal:

```text
python msma_rectangle_fem.py
```

Ein kleiner gekoppelter Funktionstest:

```text
python msma_rectangle_fem.py --smoke --no-show --out smoke_results
```

Die Eingaben stehen am Anfang von `msma_rectangle_fem.py`. Die Geometrie benötigt
nur zwei Angaben:

```python
ASPECT_RATIO = 2.0          # Breite / Höhe
MESH_DENSITY = "coarse"    # "coarse", "medium", "fine" oder ganze Zahl >= 1
```

`coarse`, `medium`, `fine` bedeuten 2, 4 bzw. 8 Elemente über die kürzere
Körperkante. Die Elementanzahl entlang der längeren Kante wird gerundet;
das Seitenverhältnis selbst bleibt exakt erhalten. Alle Elemente in Körper UND
Luft haben identische Abmessungen `dx × dy`. Sie sind rechteckig, bei allgemeinen
Seitenverhältnissen nicht zwangsläufig quadratisch. Die Körpergrenze liegt exakt
auf Elementkanten. Es gibt weder Übergangselemente noch ein zusätzliches Netzprogramm.

Eine absolute Länge muss intern festliegen: Standardhöhe 1 mm, Bezugsdicke 1 m.
Die Dickenrichtung unterliegt dem ebenen Dehnungszustand; Kräfte werden für diese
Bezugsdicke ausgegeben. Das lokale Modell hat keine innere Längenskala. Eine
gleichmäßige Skalierung von Körper und Box ändert bei gleichen Spannungs- und
Feldvorgaben daher die normierten Felder nicht, wohl aber Verschiebungen und Kräfte.

Die Luft reicht standardmäßig an jeder Seite mindestens das 1,5-Fache der
längeren Körperkante hinaus. Die Box wird auf ganze Zellen aufgerundet. Die
tatsächlichen Maße stehen in `settings.json`. `BOX_PADDING` und
`REFERENCE_HEIGHT_M` sind optionale Experteneinstellungen. Eine spätere
Boxgrößen- und Netzkonvergenzstudie ist erforderlich; der Standardabstand ist
keine nachgewiesene Näherung für das unendliche Gebiet.

## Lastpfade ohne Knotennummern

Eine `Stage` enthält Dauer, Schrittanzahl und die gewünschten ENDWERTE.
Alle angegebenen Komponenten werden innerhalb dieser Stufe gleichzeitig linear
verändert. Nicht angegebene Komponenten behalten ihren bisherigen Wert.
Mehrere Stufen erlauben sequenzielle Belastung, Halten, Entlasten und Umkehr.
Jede Komponente verläuft in jeder Stufe monoton oder konstant. Die Zeit ist bei
diesem ratenunabhängigen Modell ein Pfadparameter.

```python
USER_STAGES = [
    Stage(1.0, 10, stress_x_mpa=-1.4, name="Druck in x"),
    Stage(0.5, 10, stress_y_mpa=-0.3, mu0_h_ext_t=[0.0, 0.1],
          name="y-Druck und Magnetfeld gleichzeitig"),
    Stage(1.0, 10, mu0_h_ext_t=[0.0, 0.0], name="Feld zurücknehmen"),
    Stage(1.0, 10, stress_x_mpa=0.0, stress_y_mpa=0.0,
          name="Mechanisch entlasten"),
]
```

Die Zahlen `1.0` bzw. `0.5` sind Stufendauern, die nachfolgende ganze Zahl ist
die Anzahl der Lastinkremente. Spannungen werden in MPa vorgegeben, Zug positiv,
Druck negativ. `stress_x_mpa`, `stress_y_mpa`, `shear_mpa` definieren eine
gleichmäßige symmetrische äußere Spannung

\[
\overline\sigma=\begin{pmatrix}s_x&\tau\\\tau&s_y\end{pmatrix}.
\]

Auf jeder Körperkante wird daraus die Randtraktion
\(\boldsymbol t=\overline\sigma\boldsymbol n\) berechnet. Bei
`stress_x_mpa=-1.4` zeigt die Last auf der linken Kante nach rechts und auf der
rechten nach links. `shear_mpa` belastet beide Kantenpaare und erzeugt damit
keinen künstlichen resultierenden Starrkörpermoment.

Zusätzlich lassen sich einzelne Kanten direkt ansprechen:

```python
Stage(1.0, 10, edge_tractions_mpa={
    "left":  [0.2, 0.0],
    "right": [-0.2, 0.0],
})
```

Die Schlüssel heißen `left`, `right`, `bottom`, `top`. Die Vektoren sind immer
globale `[t_x, t_y]`, nicht Normal-/Tangentialkomponenten. Diese Zusatztraktionen
werden zu den Traktionen aus den Spannungsamplituden ADDIERT. Sie bleiben bis
zu einer neuen Vorgabe für die betreffende Kante bestehen; zum Entfernen
explizit `[0.0, 0.0]` angeben. Nicht ausgeglichene Lasten erzeugen Lagerreaktionen.

Der Code integriert die konsistenten Knotenkräfte entlang jeder Randkante:
pro Randsegment erhält jeder Endknoten `Traktion × Segmentlänge × Dicke / 2`.
Beiträge an gemeinsamen Knoten werden addiert. Es müssen keine Knotennummern
oder individuellen Knotenkräfte eingegeben werden.

## Magnetische Vorgabe und Gleichungen

`mu0_h_ext_t=[0.0,0.02]` bedeutet ein homogen angelegtes Feld mit
\(\mu_0H_{\mathrm{ext},y}=0.02\,\mathrm T\). Es ist weder die lokale Induktion
noch das bereits um die Entmagnetisierung korrigierte Feld. Intern gilt

\[
\boldsymbol h=\boldsymbol H_\mathrm{ext}-\nabla\widetilde\phi,
\qquad \widetilde\phi=0\quad\text{am äußeren Boxrand}.
\]

Die Nullrandbedingung bezieht sich auf das Entmagnetisierungspotenzial. Würde man
stattdessen das GESAMTpotenzial auf der gesamten äußeren Grenze auf null setzen,
wäre damit kein homogenes äußeres Feld vorgegeben. Alle Potential-Randknoten
sind echte Dirichlet-Freiheitsgrade; kein Strafparameter wird verwendet.

Verschiebungen existieren nur im Körper, das skalare Potenzial auf dem gesamten
Netz. Bilanzresiduen im Körper bzw. Gesamtgebiet sind

\[
R_u=\int_{\Omega_s}B^T\sigma\,dV-F,\qquad
R_\phi=\int_{\Omega}G^T(\boldsymbol b-\mu_0\boldsymbol H_\mathrm{ext})\,dV.
\]

In Luft ist \(b=\mu_0h\); im Festkörper ist \(b=\mu_0(h+m)\).
Die konstante äußere Feldkomponente wird in der Potenzialformulierung abgezogen.
Körper und Luft teilen sich ihre magnetischen Randknoten; die magnetischen
Übergangsbedingungen entstehen aus der gemeinsamen schwachen Form.

Das zugrunde liegende inkrementelle Potenzial ist, bis auf lastabhängige
Konstanten,

\[
\int_{\Omega_s}[\psi(\varepsilon,p)-\mu_0h\cdot m(p)+D(p,p_n)]\,dV
-\frac{\mu_0}{2}\int_\Omega|\nabla\widetilde\phi|^2\,dV-W_\mathrm{ext}.
\]

Es ist ein Sattelproblem hinsichtlich des magnetischen Potenzials; die lokalen
internen Variablen werden bei FESTEM \(\varepsilon,h\) minimiert. Deshalb hat
der reine magnetische Block im globalen Jacobian hier ein negatives Vorzeichen.
Ein zusätzlicher makroskopischer Entmagnetisierungstensor wird nicht verwendet.

## Material, plane strain und Zustandsverwaltung

Q4-Elemente mit 2×2-Gaußintegration liefern in jedem Körper-Integrationspunkt
`[eps_xx,eps_yy,gamma_xy]`. Die 3D-Materialroutine bekommt daraus
`[eps_xx,eps_yy,0,gamma_xy/2,0,0]` und `[h_x,h_y,0]`. Die Materialrechnung behält
die Transformationseigendehnung in z und die Reaktionsspannung `sigma_zz` bei.
Ebener Dehnungszustand bedeutet ausdrücklich NICHT `sigma_zz=0`.

Der anfängliche Zustand ist `xi1=0.5`, `eta1=eta2=0`, Winkel `theta1=theta3=0`,
`phi=pi/4`. Die Anfangsverschiebung berücksichtigt die mittlere
Transformationseigendehnung und erfüllt bei Nulllast bereits die ebenen
Spannungsbilanzen. Mit den Standardparametern bleibt aber wegen `eps_zz=0`
eine Anfangsspannung `sigma_zz=-94 MPa`. Das ist eine Folge des gewählten
Referenzzustands und der 3D-Eigendehnungen. Für einen allseitig spannungsfreien
Anfangszustand wäre eine andere Referenzdefinition oder Kinematik erforderlich.

Interne Variablen werden NUR an den vier Gaußpunkten jedes Körperelements
gespeichert. Keine internen Variablen sind globale Knotenfreiheitsgrade.
`xi2=1-xi1`; weitere Winkelzusammenhänge und die Eliminierung des Laminatvektors
`a` erfolgen in der beiliegenden Materialroutine. Sie verwendet die bereits
im homogenen Code gewählte gemeinsame isotrope Elastizität und den
Zweivarianten-Laminatansatz, kein beliebiges dreidimensionales Variantenmodell.

`state_n` bleibt während sämtlicher Newton- und Liniensuchschritte unverändert.
Ein aktueller lokaler Lösungsvektor darf als Startwert dienen, ersetzt aber
nicht den historischen Zustand. Erst ein global konvergierter Schritt wird
übernommen. Bei Fehlschlag wird der Lastschritt halbiert; konvergierte Teilschritte
werden gespeichert. Nach maximal fünf Halbierungen pro Zweig wird abgebrochen.
Ein Laufzeitfehler liefert somit keine vorgetäuschte Endlösung; bereits
konvergierte Ergebnisse bleiben verfügbar. Ein automatischer Neustart aus NPZ
ist noch nicht implementiert.

Mechanisch entfernen `u_x=u_y=0` links unten und `u_y=0` rechts unten die drei
Starrkörpermoden. Bei ausgeglichenen Lasten sind ihre Reaktionen numerisch null.
Diese drei Bindungen sind keine Einspannung einer ganzen Körperkante.

## Tangenten und künstliche Dissipation

`TANGENT_MODE="auto"` startet mit den partiellen Tangenten bei festgehaltenen
internen Variablen: `C_el`, `mu0*I`, Kopplungsblöcke null. Die Materialzustände
und die Residuen werden trotzdem bei JEDEM Aufruf vollständig aktualisiert.
Nach drei Iterationen oder einer stockenden Liniensuche werden die analytisch
kondensierten Materialtangenten einschließlich Kopplung angefordert. Bei einer
lokal ungültigen Tangente wird an diesem Punkt auf die partiellen Werte
zurückgegriffen. `"frozen"` erzwingt ausschließlich partielle Tangenten,
`"consistent"` fordert die kondensierten ab der ersten Iteration an.
Elastische Tangenten allein sind somit verfügbar, aber keine allgemeine
Konvergenzgarantie. Im getesteten magnetischen Standardlauf wurde auf
kondensierte Tangenten gewechselt.

Für die zuvor diskutierten flachen magnetischen Zustände wird standardmäßig
eine zusätzliche inkrementelle Dissipation benutzt:

\[
D_\mathrm{art}=\tfrac12\kappa_\eta
[(\eta_1-\eta_{1,n})^2+(\eta_2-\eta_{2,n})^2],
\quad \kappa_\eta=1000\,\mathrm{J/m^3}.
\]

Sie ist im FEM-Adapter implementiert, einschließlich Gradient und Hesse-Matrix.
Die ursprüngliche Materialdatei bleibt unverändert. Diese Ergänzung ÄNDERT das
inkrementelle Modell, ist nicht kalibriert und führt zu Lastschrittweitenabhängigkeit.
Sie ist kein Nachweis, dass die ursprünglichen Ausreißer ausschließlich numerisch
waren. Mit `KAPPA_ETA=0` bzw. `--kappa-eta 0` wird sie abgeschaltet. Im kleinen
magnetischen Test scheiterte der unregularisierte Lauf trotz Schrittverkleinerung
an der globalen Liniensuche; mit dem Standardwert konvergierte er. Die Größe
und der Einfluss der Regularisierung müssen für quantitative Aussagen untersucht
werden. Die bestehende Dissipation der Variantenumwandlung bleibt zusätzlich aktiv.

## Ergebnisse und GiD (erweiterte Ausgabe, Version 2)

Im Ordner `rectangle_results` entstehen:

- `rectangle.post.msh`: Q4-Netz, Materialnummer 1 = Körper, 2 = Luft.
- `rectangle.post.res`: Potenzial, Verschiebung, h, b, m, Spannungen, Dehnungen
  und interne Variablen an Knoten bzw. den vier explizit angegebenen Gaußpunkten.
- `step_XXXX.npz`: vollständige numerische Felder, Zustände und Lasten jedes
  konvergierten Schrittes. Spannungen/Dehnungen: `[xx,yy,zz,xy,yz,xz]`,
  Dehnungen mit tensorwertigem Schub; interne Zustände nur für Körperelemente,
  in aufsteigender Reihenfolge der globalen Elementnummern.
- `history.csv`: alle sechs mittleren Dehnungs- und Spannungskomponenten,
  alle drei mittleren Komponenten von m, h und b, äußere Feld- und
  Spannungsamplituden, Variantenanteil, Konvergenzmaße und Lagerreaktionen.
- `settings.json`: sämtliche Last-, Material-, Geometrie- und Solvervorgaben.
- PNG- und PDF-Diagramme: siehe unten.

Für GiD beide gleichnamigen `.post.msh`/`.post.res`-Dateien zusammenlassen und
das Ergebnisnetz im Postprozessor öffnen. Die Dateien folgen dem GiD-ASCII-
Postformat; ein Import in einer laufenden GiD-Installation wurde hier nicht
getestet. Formatreferenz:
https://gidsimulation.atlassian.net/wiki/spaces/GCM/pages/2416673648/Results+example

Mechanische Größen, Verschiebungen und interne Variablen sind in Luft nur mit
Null als Ausgabeplatzhalter belegt. Für mechanische Darstellungen in GiD deshalb
auf Material 1 filtern. h und b sind im gesamten Gebiet physikalische Ergebnisfelder.
Die NEUEN Dehnungsfelder und Körpermittelwerte werden ausschließlich für die
Körperelemente exportiert, ohne Luft-Platzhalter.

### Dehnungen und Körpermittelwerte in GiD

Unter der Analyse `MSMA` stehen jetzt `Strain xx [-]`, `Strain yy [-]` und
auch zz, xy, yz, xz bereit. Es sind absolute Gesamtdehnungen relativ zur
konstitutiven Referenz, nicht Änderungen gegenüber dem Anfang des Lastpfads.
`Strain xy` ist die tensorwertige Schubdehnung, also NICHT `gamma_xy`.

Unter der zusätzlichen Analyse `Body averages` stehen:

- `Body mean strain xx [-]` bis xz: sechs mittlere Dehnungskomponenten.
- `Body mean stress xx [Pa]` bis xz: sechs mittlere Spannungskomponenten.
- `Body mean magnetization [A/m]`: Vektor der gemittelten Magnetisierung.
- `Body mean magnetization x [A/m]`, y, z: separat wählbare Komponenten.

Für eine Größe q gilt `<q> = (1/V_body) integral_body q dV`. Beim verwendeten
regelmäßigen Netz besitzen sämtliche Körper-Gaußpunkte identische
Integrationsgewichte; deshalb ist das arithmetische Mittel hier exakt das
diskrete Volumenmittel. Luft wird NICHT mitgemittelt.

Die Körpermittelwerte werden als räumlich konstante Gaußpunktfelder auf dem
Körper gespeichert. Für einen Zeitverlauf in GiD daher einen Körperpunkt des
gewünschten Mittelwertfeldes auswählen; jeder Körperpunkt liefert denselben
Mittelwert. In NPZ stehen diese Mittelwerte zusätzlich als kurze Arrays
`mean_strain`, `mean_sigma`, `mean_m`, `mean_h`, `mean_b` bereit. Ein Mittelwert
über die Ausgabeplatzhalter der Luft wäre dagegen physikalisch falsch.

### Zusätzliche Diagramme unabhängig von GiD

Alle Diagramme werden als PNG und als vektorwertige PDF gespeichert:

| Dateibasis | Inhalt |
| --- | --- |
| `mesh` | Körper-/Luftnetz |
| `history` | Zeitverläufe der Körpermittel von Dehnungen, Spannungen, Magnetisierung, Induktion, Variantenanteil und Bilanzresiduum |
| `loading_history` | Äußere Spannungsamplituden und angelegtes Magnetfeld über der Zeit; zusätzliche Einzelkantentraktionen bleiben in settings/NPZ dokumentiert |
| `field_induced_response` | Dehnungsänderungen und mittlere Magnetisierung über den jeweils veränderten äußeren Feldkomponenten, angelehnt an Abb. 12 und 14 des Papers von 2020 |
| `stress_induced_response` | Absolute mittlere Dehnungen und Magnetisierung über den veränderten mittleren Normalspannungen, angelehnt an Abb. 13 |
| `stress_strain` | Konventionelle mittlere Spannung-Dehnung-Diagramme für xx und yy entlang des gesamten Pfads |

In den Feld-Diagrammen werden beide Normaldehnungen und beide ebenen
Magnetisierungskomponenten gezeigt, auch die quer zur Belastung. Die Abszisse
ist das ANGELEGTE Feld `mu0*H_ext`, nicht der Mittelwert des lokalen Felds.
Magnetisierungen werden im Diagramm durch `Ms` normiert; die Rohdaten in
CSV/NPZ/GiD bleiben in A/m. Dehnungen werden in den Diagrammen in Prozent,
in CSV/NPZ/GiD dimensionslos gespeichert.

Für die Felddiagramme wird von jeder Dehnung der Wert am Beginn des jeweiligen
zusammenhängenden Feldrampenabschnitts abgezogen. Mechanische Vorbelastung bei
konstantem Feld wird dabei nicht als scheinbar feldinduzierte Dehnung
mitgezeichnet. Bei gleichzeitig veränderten mechanischen Lasten enthält die
Dehnungsänderung selbstverständlich deren Einfluss ebenfalls. Absolute
Dehnungen stehen unabhängig davon in GiD, NPZ und der Zeitverlaufsdarstellung.

Die Kurven werden in zeitlicher Reihenfolge verbunden, NICHT nach dem Feld oder
der Spannung sortiert. Pfeile kennzeichnen die Laufrichtung. Belastung und
Entlastung bilden deshalb bei einem geeigneten zyklischen Pfad eine Schleife;
der Code fügt nicht eigenständig Lastzyklen hinzu. Bei unveränderter Abszisse
werden Halte-/Vorbelastungsabschnitte in den feld-/spannungsgesteuerten
Antwortdiagrammen ausgelassen; die vollständigen Zustände bleiben in den
übrigen Ausgaben erhalten. Numerische Schwankungen nahe null werden durch
eine Mindestspanne der Diagrammachse nicht optisch vergrößert; die Daten
selbst werden nicht verändert. Das Standardbeispiel mit schwachem Feld nach
abgeschlossener Variantenumwandlung zeigt praktisch keine zusätzliche
Normaldehnung, also auch keine Schmetterlingskurve. Die Diagramme sind
Auswertungen des tatsächlich berechneten Pfads, keine Reproduktion der
Paper-Ergebnisse mit abweichender Geometrie, Kinematik oder Belastung.

### Vorhandene Rechnungen nachträglich auswerten

Die NPZ-Dateien der ersten Codeversion enthalten bereits die Dehnungen.
Vorhandene Ergebnisse können daher ohne erneute FEM-Rechnung ergänzt werden:

```text
python msma_rectangle_fem.py --postprocess-only --no-show --out rectangle_results
```

Oder aus Spyder bzw. einem Python-Skript:

```python
from msma_rectangle_fem import postprocess_saved_results
postprocess_saved_results("rectangle_results", show=True)
```

Benötigt werden `settings.json`, `history.csv` und die dort aufgeführten
`step_XXXX.npz` im Ergebnisordner. GiD-Dateien, CSV und Diagramme werden neu
geschrieben; NPZ-Dateien erhalten zusätzlich die Mittelwerte. Gespeicherte
Verschiebungen, Dehnungen, Spannungen und interne Zustände werden numerisch
beibehalten. Es erfolgen KEINE lokalen Materialupdates und KEINE globalen
Gleichgewichtslösungen. Die vorhandene Schrittfolge bleibt erhalten;
überzählige alte NPZ-Dateien außerhalb von history.csv werden ignoriert.
Bei älteren Rechnungen mit einer vom Standard abweichenden Luftbox muss die
damalige Einstellung `BOX_PADDING` passen, da Version 1 sie nicht explizit
gespeichert hat. Eine abweichende rekonstruierte Geometrie führt zum Abbruch.

## Durchgeführte Prüfungen und Grenzen

`python -m unittest -v test_rectangle_fem.py` prüft zehn Fälle: identische
Elementabmessungen/exaktes Seitenverhältnis, ausgeglichene Resultierende und
Momente, konsistente Einzelkantenkräfte, Lastfortschreibung, elastisch-magnetischer
Patchtest, unbelastetes plane-strain-Gleichgewicht und magnetisches Residuum
einschließlich Feldvorzeichen/Jacobian. Dabei werden auch unverändertes
`state_n`, Randpotenziale und die Ausgabestruktur geprüft. Die Erweiterung prüft
zusätzlich Mittelwerte für nichtuniforme Gaußpunktfelder, den Ausschluss der
Luft aus Mittelwertfeldern, Hystereseabschnitte ohne Sortierung und die
nachträgliche Ausgabe ohne Materialaufruf bei unveränderten gespeicherten Feldern.

Zusätzlich wurden der gekoppelte kleine Funktionstest und eine mechanische
Variantenumwandlung bis -1.4 MPa gerechnet. Die Standardrechnung durchläuft
zuerst die Kompression und erhöht anschließend bei gehaltener Spannung das
äußere Feld in y. Die zugehörigen Beispielausgaben liegen bei.

Der Code ist ein geprüfter Forschungsprototyp, keine vollständig validierte
FEM-Implementierung. Kleine Dehnungen und feste Integrationsgebiete werden
vorausgesetzt. Die magnetische Geometrie wird nicht mitverformt; zusätzliche
Maxwell-Oberflächenlasten oder Ableitungen einer aktualisierten magnetischen
Geometrie werden nicht hinzugefügt. Q4-Elemente und das lokale nichtkonvexe
Materialmodell können Netzabhängigkeit, Zweigwechsel und Konvergenzprobleme zeigen.
Das lokale Verfahren sucht zulässige stationäre Minima und garantiert kein
globales Energieminimum. Sehr große Felder/Lastschritte sind nicht durch die
Beispieltests abgesichert. Berechnungsdauer wächst besonders mit der Zahl der
Körper-Gaußpunkte; feine Netze sollten erst nach einem groben Probelauf folgen.
