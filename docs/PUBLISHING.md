# Objavljivanje pripremljenog paketa na GitHub-u

Ovo je uputstvo za radnje koje izvrsava vlasnik naloga. Paket je napravljen
lokalno; nije kreiran udaljeni repozitorijum, nije izvrsen push i nije
objavljen licni diplomski rad.

## 1. Koristi cist paket, ne ceo razvojni folder

Otpakuj izabrani ZIP u **nov, prazan direktorijum**. U njemu treba da budu
README, aplikacija, src, dozvoljeni podaci, testovi i manifest paketa.

Ne pokreci `git add .` u starom razvojnom folderu sa licnim Word/PDF fajlovima.
Ne prenosi `.venv`, `.cache`, `.env`, pristupne tokene, fakultetske primere,
licne dokumente ili korisnicke indekse.

Gradnja iz razvojnog foldera, postojecim Python okruzenjem:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.build_release --version 0.1.0-v2-20260921-02
.\.venv\Scripts\python.exe -B -m scripts.verify_release artifacts\releases\local-search-0.1.0-v2-20260921-02.zip
```

Izlazi su direktorijum `artifacts\releases\local-search-<verzija>`, istoimeni
ZIP i `.zip.sha256`. Postojeci izlaz se nikad ne prepisuje. Ako gradnja prekine
posle kreiranja direktorijuma, on ostaje za pregled; neuspeh nije gotovo izdanje.
Ponovi sa novom oznakom tek kada utvrdis uzrok.

Provera netaknutog otpakovanog direktorijuma, **pre instalacije**:

```powershell
py -3.14 -B -m scripts.verify_release
```

Graditelj i verifikator koriste samo standardnu Python biblioteku.
Provera direktorijuma je stroga: `.venv`, `__pycache__`, novi indeks ili
izvestaji vise nisu deo izvornog paketa. Posle koriscenja proveravaj originalni
ZIP. Verifikator nikada ne raspakuje ZIP; odbija apsolutne/traversal putanje,
duple i case-colliding nazive i simbolicke linkove. Proverava allowlist-u,
velicinu i SHA-256 svih fajlova, svih 20 manifestom izabranih ulaza,
kanonske v1/v2 JSON fajlove, v2 review hash-eve i vezu zamrznutih izvestaja
sa izvornim kodom, protokolom i kopijama evaluacionog skupa.

`.sha256` je kontrolni zbir celog ZIP-a i moze se uporediti sa
`Get-FileHash -Algorithm SHA256 <zip>`. Manifest ne moze da potpisuje sam sebe:
on nije dokaz identiteta izdavaca. Sacuvaj spoljasnji kontrolni zbir uz izdanje.

### Obuhvat i izostavljeni fajlovi

Allowlist-a je u `scripts/verify_release.py`, a konkretan spisak i kontrolni
zbirovi su u `release_manifest.json` svakog paketa. Ukljuceni su aplikacija,
CSS/Streamlit konfiguracija, runtime kod, lock i dva requirements izvoza,
runtime/evaluacioni alati i odgovarajuci testovi sa helper-ima, 20 originalnih
PDF/TXT ulaza i licence, kanonski v1/v2 skupovi, prateci pregled oznaka,
originalni v1 evaluator za proveru porekla i zavrseni test/no-answer v2
eksperimenti. V2 draft ulazi su samo istorijsko poreklo pregleda, ne alternativa
kanonskom `queries_v2.json`.

Nisu ukljuceni HTML snimci i alati/testovi za njihovo autorsko pretvaranje,
processed kopije, prethodni draft/v1 eksperimenti, lokalni preparation
izvestaji/logovi sa korisnickim putanjama, modeli, indeksi, pickle fajlovi,
licni radovi, Word alati/testovi, fakultetski sabloni i prezentacije.
Istorijske reference ka izostavljenim pripremnim fajlovima u nepromenjenim
metapodacima nisu obecanje da su ti fajlovi deo izdanja.
Opcione `writing`/`browser` grupe ostaju u originalnom lockfile-u, ali nisu
potrebne za paketni runtime i unit testove.

### Lokalna provera nije nova instalacija

Paketni testovi se izvrsavaju postojecim projektnim Python-om, ali uz **cwd
unutar izdvojenog stage direktorijuma**, bez `PYTHONPATH` veze sa razvojnim
izvorima. Primer za PowerShell iz razvojnog foldera:

```powershell
$root = (Get-Location).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$stage = Join-Path $root "artifacts\releases\local-search-0.1.0-v2-20260921-02"
$testOutput = Join-Path $root "artifacts\releases\checks-0.1.0-v2-20260921-02"
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = "1"
# Putanja --basetemp mora biti namenski test direktorijum, nikad sa licnim fajlovima.
Push-Location $stage
try {
    & $python -E -B -m pytest tests\unit -q -p no:cacheprovider --basetemp $testOutput
    if ($LASTEXITCODE -ne 0) { throw "Paketni testovi nisu prosli." }
    & $python -E -B -m scripts.verify_release
} finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
}
```

Ovo proverava paketni kod i testove sa vec instaliranim zavisnostima. Nisu
ponovljeni skupa modelska evaluacija, istorijska vremena niti nezavisna
instalacija sa javnih indeksa. Dokaz ljudske validacije oznaka nije dodat.
Broj proslih testova i konkretne rezultate lokalne provere sacuvaj odvojeno od
netaknutog paketa; novo merenje ne prepisuje zamrznute rezultate.

## 2. Proveri prava i privatnost

- Procitaj [napomene o materijalima](../data/LICENSES.txt).
- MIT materijali imaju nekomercijalne i ShareAlike uslove.
- Google TXT adaptacije zadrzavaju pripisivanje i uslove CC BY 4.0.
- Citati u evaluacionom skupu nisu automatski originalni kod.
- Modelske tezine ne dodavati u repozitorijum.
- Odaberi licencu svog koda zasebno. Ovaj paket ne dodeljuje automatski MIT
  ili drugu licencu celom projektu. Samo postavljanje na GitHub ne daje
  neograniceno pravo drugima da koriste kod.

Pregled allowlist-e i poznatih rizicnih naziva nije garancija potpune bezbednosti.
Pregledaj i staged diff pre objavljivanja. Licne podatke i kredencijale ne
treba slati ni u privatni repozitorijum bez odgovarajuce potrebe i odobrenja.

## 3. Napravi repozitorijum u svom nalogu

Na GitHub-u izaberi **New repository**, zadaj naziv i opis i izaberi vidljivost.
Ako je repozitorijum privatan, profesor mora dobiti pristup. Ako je javni,
prethodno proveri sve uslove objavljivanja podataka.

Za najjednostavniji prvi push napravi prazan repozitorijum bez automatskog
README-a i dodatnih fajlova, jer paket vec ima svoju dokumentaciju.
Kopiraj stvarni HTTPS URL repozitorijuma.

## 4. Lokalni Git i prvi push

Iz PowerShell-a, u korenu **otpakovanog cistog paketa**:

```powershell
git init -b main
git status --short
git add .
git --no-pager diff --cached --stat
git --no-pager diff --cached
```

Pregledaj sta je zaista dodato. Podesi svoje stvarno ime i adresu za Git
ako ih nalog jos nema; nemoj kopirati tudji identitet.

```powershell
git commit -m "Prepare local retrieval application and frozen evaluation" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
$repoUrl = Read-Host "Unesi stvarni HTTPS URL svog novog GitHub repozitorijuma"
git remote add origin $repoUrl
git push -u origin main
```

Autentifikaciju obavi kroz GitHub-ov preporuceni mehanizam. Lozinke ili
tokene ne upisuj u README, skripte ili remote URL koji ce biti podeljen.

## 5. Provera nakon objavljivanja

Otvori repozitorijum u browseru i proveri:

1. README i slike se prikazuju.
2. Kod i evaluacioni JSON mogu da se otvore.
3. Izvori i licence su vidljivi.
4. Nema privatnih dokumenata, modelskog kesa ili tajni.
5. Instalacija i pokretanje rade iz zasebne kopije.

Sacuvaj tacan commit ili napravi oznaku izdanja, kako link u radu ne bi
neprimetno poceo da pokazuje izmenjene podatke. Tek tada u novu Word/PDF
iteraciju unesi:

- link do implementacije;
- link do `data/evaluation/queries_v2.json`;
- link do `data/full_corpus_manifest.csv` i licenci;
- po potrebi link do zamrznutih rezultata.

Do tada u radu ne treba prikazivati lazni, pretpostavljeni ili nedostupan URL.
Profesorski mejl, slanje priloga i pozivanje saradnika nisu izvrseni ovim uputstvom.
