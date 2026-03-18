.PHONY: help all amendments votes selection debates segments unique-segments debats dataset bert clean

help:
	@echo "Available targets :"
	@echo "  make amendments     	# Make {15th,16th,17th}/amendments.csv"
	@echo "  make votes        		# Make {15th,16th,17th}/votes.json"
	@echo "  make selection       	# Make export/temp/selection-{15th,16th,17th}.csv"
	@echo "  make debates         	# Make export/temp/debates/{15th,16th,17th}.csv"
	@echo "  make segments         	# Make export/temp/segments-{15th,16th,17th}.csv"
	@echo "  make unique-segments  	# Make export/temp/unique_segments.csv"
	@echo "  make debats          	# Make export/temp/debats.json"
	@echo "  make dataset         	# Make export/temp/dataset.csv"
	@echo "  make bert            	# Make export/temp/fine-tuning-bert.csv"
	@echo "  make all             	# Execute everything in the right order"
	@echo "  make clean           	# Clean all the files in export/temp/"

amendments:
	mkdir -p ./export/temp/amendments
	uv run build_amendments.py --amendments-dir ./open_data/15th/amendments/ --output-csv ./export/temp/amendments/15th.csv
	uv run build_amendments.py --amendments-dir ./open_data/16th/amendments/ --output-csv ./export/temp/amendments/16th.csv
	uv run build_amendments.py --amendments-dir ./open_data/17th/amendments/ --output-csv ./export/temp/amendments/17th.csv

votes:
	mkdir -p ./export/temp/votes
	uv run build_votes.py --votes-dir ./open_data/15th/votes/ --dossiers-dir ./open_data/15th/dossiers/ > ./export/temp/votes/15th.json
	uv run build_votes.py --votes-dir ./open_data/16th/votes/ --dossiers-dir ./open_data/16th/dossiers/ > ./export/temp/votes/16th.json
	uv run build_votes.py --votes-dir ./open_data/17th/votes/ --dossiers-dir ./open_data/17th/dossiers/ > ./export/temp/votes/17th.json

selection:
	mkdir -p ./export
	uv run scrutin_selection.py --output ./export/temp/selection-15th.csv --votes ./export/temp/votes/15th.json --amendments ./export/temp/amendments/15th.csv --dossiers ./open_data/15th/dossiers/
	uv run scrutin_selection.py --output ./export/temp/selection-16th.csv --votes ./export/temp/votes/16th.json --amendments ./export/temp/amendments/16th.csv --dossiers ./open_data/16th/dossiers/
	uv run scrutin_selection.py --output ./export/temp/selection-17th.csv --votes ./export/temp/votes/17th.json --amendments ./export/temp/amendments/17th.csv --dossiers ./open_data/17th/dossiers/

debates:
	mkdir -p ./export/temp/debates
	uv run build_debates.py --dir-debates ./open_data/15th/debates/ --output-csv ./export/temp/debates/15th.csv
	uv run build_debates.py --dir-debates ./open_data/16th/debates/ --output-csv ./export/temp/debates/16th.csv
	uv run build_debates.py --dir-debates ./open_data/17th/debates/ --output-csv ./export/temp/debates/17th.csv

segments:
	uv run segments.py --votes-json ./export/temp/votes/15th.json --debates-csv ./export/temp/debates/15th.csv --votes-selection-csv ./export/temp/selection-15th.csv --output-csv ./export/temp/segments-15th.csv
	uv run segments.py --votes-json ./export/temp/votes/16th.json --debates-csv ./export/temp/debates/16th.csv --votes-selection-csv ./export/temp/selection-16th.csv --output-csv ./export/temp/segments-16th.csv
	uv run segments.py --votes-json ./export/temp/votes/17th.json --debates-csv ./export/temp/debates/17th.csv --votes-selection-csv ./export/temp/selection-17th.csv --output-csv ./export/temp/segments-17th.csv

unique-segments:
	xan cat rows ./export/temp/segments-15th.csv ./export/temp/segments-16th.csv ./export/temp/segments-17th.csv | \
		xan search -e 'False' -s 'drawer' | \
		xan search -e 'True' -s found | \
		xan search -e 'False' -s 'rappel_reglement' > ./export/temp/unique_segments.csv

debats:
	uv run regroupement.py --segments-csv ./export/temp/unique_segments.csv --output-json ./export/temp/debats.json

dataset:
	uv run dataset.py --input ./export/temp/debats.json --output ./export/dataset.csv

bert:
	uv run bert_dataset.py --csv-15th-debates ./export/temp/debates/15th.csv --csv-16th-debates ./export/temp/debates/16th.csv --csv-17th-debates ./export/temp/debates/17th.csv --csv-15th-amendments ./export/temp/amendments/15th.csv --csv-16th-amendments ./export/temp/amendments/16th.csv --csv-17th-amendments ./export/temp/amendments/17th.csv --output ./export/fine-tuning-bert.csv

all: amendments votes selection debates segments unique-segments debats dataset bert

clean:
	rm -f ./export/temp/selection-15th.csv ./export/temp/selection-16th.csv ./export/temp/selection-17th.csv
	rm -f ./export/temp/debates/15th.csv ./export/temp/debates/16th.csv ./export/temp/debates/17th.csv
	rm -f ./export/temp/segments-15th.csv ./export/temp/segments-16th.csv ./export/temp/segments-17th.csv
	rm -f ./export/temp/unique_segments.csv ./export/temp/debats.json ./export/dataset.csv ./export/fine-tuning-bert.csv
