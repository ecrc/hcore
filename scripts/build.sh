#!/bin/bash -le



# BASH verbose mode
set -x 
currdir=$PWD

echo "Current dir is $PWD. The files in the current dir are here:"; ls -al
if [ -z $reponame ]; then reponame=hcore-dev; fi
echo "Reponame is: $reponame"

# Check if we are already in hicma repo dir or not.
if git remote -v | grep -q "https://github.com/ecrc/$reponame"
then
	# we are, lets go to the top dir (where .git is)
	until test -d $PWD/.git ;
	do
		cd ..
	done;
else
	#we are not, we need to clone the repo
	git clone https://github.com/ecrc/$reponame.git
	cd $reponame
fi
module purge
if [ "$HOSTNAME" == "thana" ]; then
	. ./scripts/power8.modules
else
    echo "Loading intel modules"
	. ./scripts/modules-ecrc.sh
fi
module list

# Update submodules
HICMADEVDIR=$PWD 
git submodule update --init --recursive

# HCORE
cd $HICMADEVDIR
rm -rf build
mkdir -p build/installdir
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/installdir
make clean
make -j
make install
ctest -T Test --no-compress-output -V
export PKG_CONFIG_PATH=$PWD/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

cd $currdir
set +x
