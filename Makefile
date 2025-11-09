.ONESHELL:
.SHELLFLAGS = -ec
.SILENT:
version := $(shell python -c "import tomllib;print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

.PHONY: dist
dist:
	python -m build

.PHONY: release
release: lock dist
	# check that changelog is updated.  only look at first 3 parts of semver
	version=$(version)
	stripped=$$(echo $${version} | cut -d . -f -3 | cut -d '-' -f 1)
	if ! grep $${stripped} CHANGELOG.rst
	then
		echo "Changelog doesn't seem to be updated! Quitting..."
		exit 1
	fi
	# generate release notes from changelog
	awk "BEGIN{p=0}; /^$${stripped}/{next}; /---/{p=1;next}; /^$$/{exit}; p {print}" CHANGELOG.rst > TMPNOTES
	# make github and pypi release
	gh release create --latest --verify-tag v$(version) dist/tomosphero-$(version)* -F TMPNOTES
	twine upload -u __token__ dist/tomosphero-$(version)*
	rm TMPNOTES

.PHONY: lock
lock:
	# run tests then make a requirements.txt lockfile
	rm -rf .venv_lock
	virtualenv .venv_lock
	. .venv_lock/bin/activate
	pip install .[test]
	pytest tomosphero
	pip freeze > requirements.txt

.PHONY: tag
tag:
	# tag git commit
	git add requirements.txt
	git add pyproject.toml
	git add CHANGELOG.rst
	git commit -m "bump version" --allow-empty
	git tag -a v$(version) -m "version $(version)"
	git push --tags
	git push

.PHONY: docs
docs:
	pdoc -o docs \
		--docformat google --no-search tomosphero \
		'!tomosphero.test_' \
		-t template/ \
		--footer-text "tomosphero ${version}" \
		--math \
		--logo "https://evan.widloski.com/tomosphero/logo.svg"
	touch docs/.nojekyll
	ghp-import -f -p -b docs docs

.PHONY: joss
joss:
	podman run -it \
		--volume ./:/data \
		--env JOURNAL=joss \
		docker.io/openjournals/inara