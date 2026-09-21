# Korpus i evaluacioni skup

## Sta se pretrazuje

[Manifest](../data/full_corpus_manifest.csv) navodi 20 zamrznutih materijala:
tri originalna MIT PDF-a i 17 tekstualnih prilagodjavanja Google lekcija.
Za svaki ulaz sacuvani su autor, URL, licenca, transformacija i SHA-256.
U paketu se nalaze upravo fajlovi na putanjama `relative_path`, ne kopije
korisnikovog proizvoljnog indeksa.

Google TXT verzije zadrzavaju napomenu o poreklu. Izvorni HTML snimci ostaju
u razvojnoj arhivi, ali nisu potrebni za indeksiranje zamrznutog korpusa i
nisu ukljuceni u cist paket. Kolona `source_relative_path` u manifestu je
istorijski podatak o tim snimcima, ne obecanje da su svi HTML fajlovi deo paketa.
Isto vazi za istorijske reference na `artifacts/preparation/ingestion_report.json`:
originalni lokalni izvestaj sadrzi korisnicke putanje i nije deo cistog izdanja.

Ne preuzimati danasnje stranice preko postojecih TXT fajlova pa tvrditi da
je korpus ostao isti. Sadrzaj javnih stranica moze da se promeni.

PDF tekst ima poznate probleme: tri upozorenja o xref strukturi i 457
zamenskih znakova pri ekstrakciji. Nema OCR-a ni automatskog izmisljanja
izgubljenih simbola. Koristi se ista podela za sve tri metode:
prozor oko 200 modelskih tokena, preklapanje 40, bez prelaska PDF strane.
Rezultat je 226 odlomaka.

## V2 pitanja i oznake

[queries_v2.json](../data/evaluation/queries_v2.json) sadrzi:

- 10 razvojnih pitanja;
- 100 test formulacija, od kojih 40 potice iz v1 i 60 je dodato;
- 5 odvojenih pitanja bez odgovora;
- 95 oznacenih test informacionih potreba.

SHA-256 skupa:

```text
5c312d014a988e1396518d95546ca486e919d9a4ddc3b4b4150b9a24bbd243bf
```

Za svako pitanje nalaze se tekst, tema, tip formulacije, grupa potrebe,
pozitivni chunk ID-jevi, citati i pojedinacne odluke o pregledanim kandidatima.
Jedan odlomak mora samostalno dati dovoljno informacija za odgovor.
Pominjanje teme ili prekinuta definicija nisu automatski dovoljan dokaz.

Pregledana je unija top-5 kandidata sve tri metode i pocetnih predloga
pozitivnih odlomaka, bez dupliranja unutar istog pitanja: **1039 parova**,
od kojih 210 relevantnih i 829 nerelevantnih. Tri nejasnoce dodatno su
razresene pre zamrzavanja.

Pregled je obavio asistent. `human_review_complete=false` i
`review_provenance.source=assistant` namerno ostaju u metapodacima.
To nije nezavisna ljudska ili ekspertska validacija.

## Sta mere rezultati

- Hit@5: bar jedan relevantan odlomak u prvih pet.
- MRR@5: prosek reciprocnih rangova prvog relevantnog odlomka, nula za promasaj.
- Precision@5: broj relevantnih medju prvih pet, podeljen sa pet, pa prosek preko pitanja.

Vremenska ponavljanja ne povecavaju broj nezavisnih pitanja. No-answer pitanja
ne ulaze u ove metrike. Oznake pokrivaju odabrani pool, ne sve moguce parove
pitanja i odlomaka. Nova metoda moze zahtevati nove odluke.

Originalne formulacije i svih 600 njihovih ranijih rangiranih pozicija
ostali su isti, ali su pozitivni skupovi 11 originalnih test pitanja ponovo
procenjeni. Zato promene v1/v2 kvaliteta nisu cisto poboljsanje algoritama.

## Proverljivost i licence

[Protokol](../config/evaluation_v2.json), [rezultati](../artifacts/experiments/test_three_methods_v2/analysis.json)
i sirovi CSV fajlovi omogucavaju proveru brojki. Identifikatori odlomaka
ponovo se dobijaju iz zamrznutih ulaza, istog koda i modela. Novi indeks
ima novi generacijski ID; on se ne prepravlja radi uklapanja u stari izvestaj.

Materijali i citirani odlomci zadrzavaju izvorne licence:
[data/LICENSES.txt](../data/LICENSES.txt). Ne primenjuj licencu aplikacionog
koda na MIT/Google sadrzaj kao da ga je autor aplikacije napisao. Modelske
tezine nisu ukljucene; [njihov status](../config/model_provenance.json) ostaje odvojen.
