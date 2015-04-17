REPO_URL=https://github.com/computationalmodelling/pytest_validate_nb.git
INSTALL_DIR=tmp-install-pytest_validate_nb
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR
if [ -e pytest_validate_nb ]; then
    echo "Found git repository, try updating..."
    cd pytest_validate_nb
    git pull
else
    echo "Cloning git repository"
    git clone https://github.com/computationalmodelling/pytest_validate_nb.git
    cd pytest_validate_nb
fi
COMMAND="sudo pip install -U ."
echo "About to execute: sudo $COMMAND"
$COMMAND

