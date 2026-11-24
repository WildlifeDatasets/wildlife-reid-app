# Implementační plán: uložené návrhy identity a sjednocení úložiště

První čtyři etapy jsou implementované. Etapa 4 přidává volitelný širší výběr
referencí, ruční seskupování neznámých observací a kontrolu známých přiřazení.
Ověření kvality na reálných datech, úplné ověření na GPU a nasazení zůstávají otevřené.

## Cíl a hranice

- [x] Prověřit současné ukládání návrhů, obrazovku identifikace a databázovou vrstvu workeru.
- [x] Zapsat tento implementační plán.
- [x] Etapa 1: otevírat existující identifikační UI nad uloženými
  `MediafileIdentificationSuggestion`, bez synchronního prohledávání galerie.
- [x] Etapa 2: bezpečně navázat návrhy na jednotlivé observations a jejich vstupní verze.
- [x] Etapa 3: sjednotit úložiště příznaků při zachování identifikační metody a minimálních změnách workeru.
- [x] Etapa 4: volitelně rozšířit reference a umožnit návrhy mezi neznámými observations.

Priorita: nejprve zprovoznit čtení toho, co už máme. V etapě 1 se nemění
`identification_worker`, algoritmus identifikace, generování embeddingů ani databázové schéma.
Součástí tohoto úkolu není automatické shlukování, změna WildFusion na cosine,
implementace tracků nebo automatické přiřazování identit novým dialogem.

## Co již existuje

- `MediafilesForIdentification` odkazuje na media file.
- `MediafileIdentificationSuggestion` ukládá kandidátní identitu, referenční media file,
  skóre a párové body. Cílová observation se nyní určuje přes
  `mediafile.metadata_json["reid_observation_id"]`.
- `get_individual_identity_from_foridentification()` a šablony
  `get_individual_identity*.html` zobrazují uložené kandidáty, reference a zoom.
- `tasks.py` při ukládání identifikace vytváří návrhy pro ruční kontrolu jen v příslušné
  větvi podle prahu skóre. Výsledky nad prahem se mohou přiřadit automaticky;
  existence embeddingu proto nezaručuje existenci řádku s návrhy.
- Ukládání dalšího výsledku pro tentýž media file nyní přepisuje jeho návrhy.
  Metadata proto nejsou plnohodnotným uložištěm více nezávislých výsledků pro více bboxů.
- `MergeIdentitySuggestionResult` řeší slučování podle jmen/kódů;
  `IdentificationSimilarPairResult` hledá podobné již identifikované identity.
  Ani jedno není obecnou tabulkou kandidátů pro libovolnou neznámou observation.

## Etapa 1 — existující návrhy a existující UI; worker beze změn

### Výběr cílové observation a otevření obrazovky

- [x] Nahradit akci nového cosine dialogu akcí otevírající existující
  `get_individual_identity` (uživatel jej může znát jako `get_identification`).
  První verze může být samostatná stránka; nevkládat celou stránku do dalšího modalu.
- [x] Vstup přijímá explicitní UUID observation a server ověřuje přístup i workgroup.
- [x] Dohledat uloženou položku identifikace a ověřit její vazbu na cílovou observation.
  Existuje-li `reid_observation_id`, musí odpovídat a observation musí v media file stále existovat.
- [x] Legacy fallback bez `reid_observation_id` připustit pouze při právě jedné skutečné
  observation v media file a bez rozporu s dostupnými metadaty.
  Neplatné či cizí ID nesmí spadnout do fallbacku.
- [x] U více bboxů s nejednoznačnou vazbou zobrazit, že návrhy pro vybraný objekt nejsou
  dostupné. Nepoužít návrhy sourozence a nevybírat automaticky první observation.
- [x] Jasně uvést, že jde o poslední uložený výsledek identifikace. Legacy výsledky
  bez fingerprintu nelze zpětně prohlásit za ověřeně aktuální.
- [x] Ověřit současnou invalidaci výsledků při změnách bboxu/zdroje/orientace/modelu.
  Kde invalidace není prokazatelná, výsledek označit jako neověřený nebo jej nepoužít;
  při otevření nekontrolovat soubory celé galerie.
- [x] Při chybějících návrzích vrátit krátkou odpověď s vysvětlením a odkazem na
  existující spuštění identifikace. V této etapě nepřidávat nový výpočet jednoho bboxu.
- [x] Před otevřením z editoru vyžadovat uložení změněných vstupů; nový neuložený bbox
  nemá mít akci pro čtení uloženého výsledku.

### Použití stávajícího UI a potvrzení

- [x] Zachovat stávající porovnávací karty, procházení referencí, zoom a párové body.
  Zkontrolovat, zda náhled správně označuje cílový objekt při více observations.
- [x] Předat explicitní cílovou observation od vstupní akce až po potvrzení.
  Nepoužívat sdílenou změnu `reid_observation_id` jako stav otevřeného okna.
- [x] Doplnit návrat do Observations nebo editoru se zachováním filtrů a kotvy;
  návratovou URL ověřit jako lokální. Zachovat i původní průchod frontou identifikace.
- [x] Potvrzení identity provádět přes CSRF chráněný POST. Prověřit současné
  `set_individual_identity()`, které používá GET, a převést související odkazy/formuláře.
- [x] Potvrzení smí změnit právě jednu skutečnou observation. Ověřit skupinu cíle
  i identity, zaznamenat autora/čas a odmítnout přepsání mezitím změněné identity/vstupu.
- [x] Při změně identity správně vyřešit příznak reprezentativnosti.
- [x] Při úklidu položky staré fronty nesmazat výsledek, který mezitím vznikl pro jinou
  observation téhož media file. Při nejednoznačnosti výsledek nemazat automaticky.
- [x] Omezit přípravu stránky na potřebné kandidáty a zachovat líné načítání ostatních
  referencí. Prověřit opakované dotazy a generování náhledů během HTTP požadavku.

### Co z poslední implementace odstranit nebo převést

Odstraňovat až po přepojení všech vstupů a ověření, že se zachovala jejich funkce.

- [x] Odstranit `services/identity_candidates.py`: synchronní `search()`, normalizaci
  a cosine scan, průchod historickými vektory, výběr profilu a cache SHA-256 zdrojů.
  Pokud bude později potřeba verzování vstupu, využít společný mechanismus;
  neponechávat tuto službu jen kvůli pomocné funkci.
- [x] Zrušit GET endpoint `/observations/<uuid>/identity-candidates/` a jeho obsluhu.
  Akci nahradí krátký vstup do existujícího UI.
- [x] Sloučit potřebné pojistky z `views_identity_candidates.py` do společného
  potvrzovacího workflow a poté odstranit duplicitní view/endpoint pro přijetí kandidáta.
- [x] Odstranit `static/caidapp/js/identity-candidates.js`, nový modal
  `includes/identity_candidates_modal.html` a jejich globální registraci v `base.html`.
- [x] Převést `includes/identity_candidate_button.html` na vstup do stávajícího UI;
  odstranit atributy pro nový vyhledávací endpoint a generování embeddingu v dialogu.
- [x] Odstranit polling a generování embeddingů přidané výhradně pro nový dialog.
  Zachovat samostatné generování embeddingů v Observations/editoru a stránku Tasks.
- [x] Nahradit testy zrušeného cosine scanu regresními testy ve stávající sadě
  pro čtení uložených návrhů a potvrzení identity.
- [x] Aktualizovat nebo nahradit `docs/observation-identity-candidates.md`.
- [x] Zachovat `ObservationEmbedding`, `EmbeddingProfile`, embedding runs,
  generování embeddingů a existující starší identifikační pipeline.

### Ověření a dokončení etapy 1

- [x] Testy: existující i chybějící návrhy, více bboxů, rozporná metadata,
  odstraněná observation a cizí workgroup.
- [x] Testy: POST/CSRF, změněný cíl uloženého výsledku a zachování sourozenců.
- [x] Ověřit vykreslení uložených návrhů, zoom, ruční potvrzení, návrat a původní
  průchod frontou v regresních testech. V Edge ověřit navigaci editoru a ochranu
  neuložených změn na simulované stránce.
- [ ] Ověřit celý průchod z karty, řádku a editoru v prohlížeči na nasazené instanci.
- [ ] Ověřit na velké galerii, že otevření nečte embeddingy ani zdrojové soubory celé galerie
  a nespouští nový výpočet. Změřit odezvu a počet dotazů.
- [x] Potvrdit, že diff v `src/identification_worker/` je prázdný.

## Etapa 2 — návrhy jako výsledek pro konkrétní observation

- [x] Zavést minimální migraci současných modelů s explicitní cílovou observation
  u položky/výsledku identifikace a podle potřeby referenční observation u návrhu.
  Nezavádět druhou konkurenční tabulku kandidátů bez důvodu.
- [x] Připravit migraci `0188`, která vyčistí stávající položky čekající na kontrolu a jejich návrhy.
  Staré vazby přes media file ani `reid_observation_id` se nepřevádějí.
  Zachovat observations, přiřazené identity a uložené příznaky a embeddingy.
- [x] Ověřit migraci `0188` v lokální PostgreSQL s naplněnou frontou a vyzkoušet
  regeneraci přes skutečný identification worker.
- [ ] Po nasazení mimo lokální vývoj znovu vygenerovat návrhy pro způsobilé observations ve všech
  workgroups existujícím asynchronním workflow. Výpočet spustit až po nasazení
  nového ukládání výsledků příkazem `python manage.py regenerate_identity_suggestions`.
  Příkaz dává všechny výsledky do fronty k ruční kontrole.
- [x] V testovací DB ověřit samostatné uložení návrhů pro dva bboxy a potvrdit,
  že vyčištění staré fronty zachovalo přiřazenou identitu a uložený embedding.
- [x] Doplnit verzi vstupu, identifikační model, původ výsledku a datum uložení.
  Rozlišit aktuální, zastaralý a chybějící výsledek. Neúspěch běhu je v přehledu identifikace.
- [x] Změnit API callback tak, aby ukládání výsledku jednoho bboxu nepřepisovalo výsledky
  dalších observations téhož media file. Využít již přenášené ID observation tam, kde je dostupné.
- [x] Prověřit callbacky, zoom a potvrzení proti více observations
  ve stejném media file; doplnit testy idempotence a výsledků doručených po změně vstupu.
- [ ] Pokud bude potřeba nový výpočet jedné observation, spustit existující asynchronní
  identifikační úlohu s přesným výběrem a dohledatelným stavem. Žádné dlouhé HTTP hledání.

Worker: cílem zůstává 0 změn; pokud některá větev nezachovává identifikátor observation,
omezit zásah na jeho přenos ve vstupu/výstupu. Neměnit porovnávací algoritmus.

## Etapa 3 — společné příznaky pro jednotlivé observations

- [x] Zvolit aplikační PostgreSQL jako zdroj příznaků a ukládat pro každou způsobilou
  observation globální vektor společně s ALIKED (`keypoints`, `descriptors`,
  `image_size`). Stejná verze příznaků se použije pro dotaz i referenci.
- [x] V první verzi porovnávat observations bez identity s identifikovanými
  reprezentativními observations ve stejné pracovní skupině. Reprezentativnost
  patří k výběru reference. Nastavení příznaků ani jejich platnost na ní nezávisí.
- [x] Uložit kompletní sadu jako verzovaný výsledek observation v API. Evidovat verzi
  bboxu a zdroje, modelu a vah, segmentace, transformací i formátu příznaků.
  Využít společné verzování s `ObservationEmbedding`, pokud pasuje na celý formát.
  Existující globální vektory lze použít až po ověření kompatibility předzpracování.
- [x] Využít stávající `encode_images()` pro výpočet obou částí. Rozšířit úlohu
  pro generování příznaků, aby přijímala libovolné skutečné obrazové observations
  s uloženým bboxem bez ohledu na identitu či reprezentativnost. Pro skutečné observations
  bez bboxu použít celý obrázek. Videa vyžadují jednoznačný snímek nebo čas.
- [x] Ukládat odvozený výřez pod klíčem observation a verze vstupu, aby si dva
  bboxy jednoho media file nepřepsaly výřez. API ověří aktuálnost vstupu a profilu
  před uložením výsledku.
- [x] Ponechat `init identification` jako akci pro přípravu referencí. Vybere
  reprezentativní observations, nechá spočítat jen chybějící či zastaralé sady
  a dokončí inicializaci po jejich uložení. Změna identity nebo reprezentativnosti
  změní výběr reference, existující příznaky zůstávají použitelné.
- [x] Pro `identify` připravit na straně API verzi seznamu dotazů a referencí.
  Worker načte jen příznaky z tohoto seznamu. API exportuje data z aplikační
  PostgreSQL do souborů s pevným pořadím pro adaptér v `utils/database/`.
  API spravuje výběr, zápisy a migrace. Worker při porovnání nepotřebuje přístupové
  údaje do Django DB.
- [x] Přepojit plnou i dávkovou větev `identify` na načtení uložených příznaků pro
  obě strany. Pokud dotazu nebo referenci chybí kompatibilní sada, API spustí její
  výpočet mimo HTTP požadavek nebo oznámí chybějící pokrytí. Porovnání zachová
  WildFusion, LightGlue, kalibraci a párové body.
- [x] Vrátit ID cílové i odpovídající referenční observation. V UI zobrazit jako
  kandidáta známou identitu a její nejlépe odpovídající referenční bbox. Cesta k
  souboru slouží k náhledu, ID k přiřazení výsledku.
- [x] Ověřit lokální případ dvou bboxů téhož media file na obou stranách porovnání,
  změnu reprezentativnosti bez přepočtu, změnu bboxu s přepočtem, chybějící
  příznaky a pořadí referencí v dávkách pomocí testů.
- [ ] Ověřit shodu plné a dávkové větve a výkon při běhu se skutečnými modely na GPU.
- [x] Přepojit outliery a podobné páry na aplikační příznaky. Trénování už
  dostává vstupy z API. Odstranit samostatnou databázi workeru z Compose.
- [ ] Po nasazení přepočítat příznaky s profilem `wildfusion-v2` a obnovit návrhy.

Zásah do workeru se soustředí na společný encoder, verzované cesty výřezů,
načtení příznaků v `predict_full()` a dávkové větvi a předání ID reference.
Django rozhoduje, které observations se porovnávají. Porovnávací metoda a formát
globálních a lokálních příznaků zůstávají použitelné.

## Etapa 4 — další režimy porovnání

| Režim | Dotazy | S čím je porovnat | Výsledek |
| --- | --- | --- | --- |
| Přiřazení identity (etapa 3) | Observations bez identity | Známé reprezentativní observations | Návrhy identit s přesným referenčním bboxem |
| Neznámý–neznámý | Observations bez identity | Jiné observations bez identity | Dvojice k ručnímu seskupení |
| Kontrola přiřazení | Observations se známou identitou | Jiné známé observations ve zvolené galerii | Shody a rozpory k ruční kontrole |

- [x] Přidat možnost porovnávat se všemi identifikovanými observations, tedy i s
  nereprezentativními. V každém režimu vyloučit porovnání observation se sebou.
- [x] Pro režim neznámý–neznámý vracet top-k jednotlivých observations. Všechna
  neznámá data nesmějí sdílet zástupný `class_id`, který by je při výběru sloučil.
- [x] Po ručním potvrzení umožnit přiřazení do samostatné dočasné identity
  (`Unknown-0042` apod.) s pozdějším pojmenováním. Řetězec podobností A–B a B–C
  slouží k revizi kandidátů. Potvrzená identita se eviduje na observations.
- [x] Při kontrole známých porovnat nejlepší jinou referenci stejné identity s
  nejlepší referencí jiné identity. Výsledek uložit jako podnět ke kontrole.
  Potvrzené přiřazení zůstává na uživateli.
- [x] Použít současné uložené příznaky ve všech režimech. Seskupování a kontrola
  přiřazení řadí dvojice podle kosinové podobnosti globálních embeddingů.
- [ ] Vyhodnotit kvalitu každého režimu na známých případech.

Přehled Tasks zobrazuje inicializaci, identifikaci, outliery, návrhy na sloučení
a běhy porovnávání observací. Běžná revize identity zobrazuje jen observace bez
identity. Potvrzení outlieru cílí na konkrétní verzi jedné observace.

## Hlavní soubory při realizaci

API: `src/api/caidapp/views.py`, `urls.py`, `tasks.py`, `models.py`,
`views_mediafile.py`, šablony `get_individual_identity*.html`,
`includes/observation_card.html`, `includes/observation_list_row.html`,
`media_file_update.html` a příslušné testy.

Úklid posledního dialogu: `views_identity_candidates.py`,
`services/identity_candidates.py`, `static/caidapp/js/identity-candidates.js`,
`includes/identity_candidate_button.html`, `includes/identity_candidates_modal.html`,
`templates/base.html`, testy a dokumentace uvedené výše.

Worker až v etapě 3: `src/identification_worker/utils/database/{connection,services,model,__init__}.py`,
případně malé napojení a mapování metadat v `worker.py`. `utils/inference_identification.py`
má v minimální variantě zůstat algoritmicky beze změny.
