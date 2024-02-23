if [[ ! -d joern-cli ]]; then
    curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" -o joern-install.sh
    chmod u+x joern-install.sh
    ./joern-install.sh
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi