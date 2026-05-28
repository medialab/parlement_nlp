.PHONY: download build_open_data parse align dataset paires-triplets all

count15 := $(shell xan count ./processed/debates/15th.csv)
count16 := $(shell xan count ./processed/debates/16th.csv)
count17 := $(shell xan count ./processed/debates/17th.csv)

URL_ACTORS="https://data.assemblee-nationale.fr/static/openData/repository/17/amo/tous_acteurs_mandats_organes_xi_legislature/AMO30_tous_acteurs_tous_mandats_tous_organes_historique.json.zip"

URL_SCRUTINS_15E="http://data.assemblee-nationale.fr/static/openData/repository/15/loi/scrutins/Scrutins_XV.json.zip"
URL_SCRUTINS_16E="http://data.assemblee-nationale.fr/static/openData/repository/16/loi/scrutins/Scrutins.json.zip"
URL_SCRUTINS_17E="http://data.assemblee-nationale.fr/static/openData/repository/17/loi/scrutins/Scrutins.json.zip"

URL_DEBATES_15E="https://data.assemblee-nationale.fr/static/openData/repository/15/vp/syceronbrut/syseron.xml.zip"
URL_DEBATES_16E="https://data.assemblee-nationale.fr/static/openData/repository/16/vp/syceronbrut/syseron.xml.zip"
URL_DEBATES_17E="https://data.assemblee-nationale.fr/static/openData/repository/17/vp/syceronbrut/syseron.xml.zip"

URL_AMENDMENTS_15E="https://data.assemblee-nationale.fr/static/openData/repository/15/loi/amendements_legis/Amendements_XV.json.zip"
URL_AMENDMENTS_16E="https://data.assemblee-nationale.fr/static/openData/repository/16/loi/amendements_div_legis/Amendements.json.zip"
URL_AMENDMENTS_17E="https://data.assemblee-nationale.fr/static/openData/repository/17/loi/amendements_div_legis/Amendements.json.zip"

URL_DOSSIERS_15E="https://data.assemblee-nationale.fr/static/openData/repository/15/loi/dossiers_legislatifs/Dossiers_Legislatifs_XV.json.zip"
URL_DOSSIERS_16E="https://data.assemblee-nationale.fr/static/openData/repository/16/loi/dossiers_legislatifs/Dossiers_Legislatifs.json.zip"
URL_DOSSIERS_17E="https://data.assemblee-nationale.fr/static/openData/repository/17/loi/dossiers_legislatifs/Dossiers_Legislatifs.json.zip"

download:
	mkdir -p temp open_data/15th open_data/16th open_data/17th open_data/actors
	
	curl $(URL_ACTORS) -L -o temp/actors.zip
	unzip temp/actors.zip -d temp/actors
	mv temp/actors/json/acteur open_data/actors/acteur
	mv temp/actors/json/organe open_data/actors/organe

	curl $(URL_SCRUTINS_15E) -L -o temp/scrutins_15th.json.zip
	curl $(URL_SCRUTINS_16E) -L -o temp/scrutins_16th.json.zip
	curl $(URL_SCRUTINS_17E) -L -o temp/scrutins_17th.json.zip

	unzip temp/scrutins_15th.json.zip -d open_data/15th/
	unzip temp/scrutins_16th.json.zip -d open_data/16th/
	unzip temp/scrutins_17th.json.zip -d open_data/17th/

	mv open_data/15th/{json,votes}
	mv open_data/16th/{json,votes}
	mv open_data/17th/{json,votes}

	curl $(URL_DEBATES_15E) -L -o temp/debates_15th.zip
	curl $(URL_DEBATES_16E) -L -o temp/debates_16th.zip
	curl $(URL_DEBATES_17E) -L -o temp/debates_17th.zip

	unzip temp/debates_15th.zip -d temp/debates_15th
	unzip temp/debates_16th.zip -d temp/debates_16th
	unzip temp/debates_17th.zip -d temp/debates_17th
	
	mv temp/debates_15th/xml/compteRendu open_data/15th/debates
	mv temp/debates_16th/xml/compteRendu open_data/16th/debates
	mv temp/debates_17th/xml/compteRendu open_data/17th/debates

	curl $(URL_DOSSIERS_15E) -L -o temp/dossiers_15th.zip
	curl $(URL_DOSSIERS_16E) -L -o temp/dossiers_16th.zip
	curl $(URL_DOSSIERS_17E) -L -o temp/dossiers_17th.zip

	unzip temp/dossiers_15th.zip -d temp/dossiers_15th
	unzip temp/dossiers_16th.zip -d temp/dossiers_16th
	unzip temp/dossiers_17th.zip -d temp/dossiers_17th

	mv temp/dossiers_15th/json/dossierParlementaire open_data/15th/dossiers
	mv temp/dossiers_16th/json/dossierParlementaire open_data/16th/dossiers
	mv temp/dossiers_17th/json/dossierParlementaire open_data/17th/dossiers

	rm -r temp

build_open_data:
	mkdir -p processed processed/debates processed/votes

	uv run build_debates.py --dir-debates ./open_data/15th/debates --output-csv ./processed/debates/15th.csv
	uv run build_debates.py --dir-debates ./open_data/16th/debates --output-csv ./processed/debates/16th.csv
	uv run build_debates.py --dir-debates ./open_data/17th/debates --output-csv ./processed/debates/17th.csv

	uv run build_votes.py --votes-dir ./open_data/15th/votes --dossiers-dir ./open_data/15th/dossiers > ./processed/votes/15th.json
	uv run build_votes.py --votes-dir ./open_data/16th/votes --dossiers-dir ./open_data/16th/dossiers > ./processed/votes/16th.json
	uv run build_votes.py --votes-dir ./open_data/17th/votes --dossiers-dir ./open_data/17th/dossiers > ./processed/votes/17th.json

parse:
	uv run parse.py ./processed/debates/15th.csv | xan progress --total $(count15) > ./processed/15th_parsed.csv
	uv run parse.py ./processed/debates/16th.csv | xan progress --total $(count16) > ./processed/16th_parsed.csv
	uv run parse.py ./processed/debates/17th.csv | xan progress --total $(count17) > ./processed/17th_parsed.csv

align:
	uv run align.py --input ./processed/15th_parsed.csv --votes ./processed/votes/15th.json | xan progress --total $(count15) > ./processed/15th_aligned.csv
	uv run align.py --input ./processed/16th_parsed.csv --votes ./processed/votes/16th.json | xan progress --total $(count16) > ./processed/16th_aligned.csv
	uv run align.py --input ./processed/17th_parsed.csv --votes ./processed/votes/17th.json | xan progress --total $(count17) > ./processed/17th_aligned.csv

dataset:
	uv run dataset.py ./processed/*th_aligned.csv | xan filter 'len(speech) < 40000' > ./processed/dataset.csv

paires-triplets:
	uv run paires_triplets.py ./processed/dataset.csv ./processed/paires.csv ./processed/triplets.csv

all: download build_open_data parse align dataset paires-triplets

