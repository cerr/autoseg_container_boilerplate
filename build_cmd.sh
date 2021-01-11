#!/bin/bash
#
# File:
#	build_cmd.sh
#
# Description: 
#	basic build script for SIF container
# 	must be run with admin privliege
#	adds repo name and commit hash to filename if run from within a git folder
#
# Input Arguments:
# 	SINGRECIPE: This is the path to the recipe file (can be either relative or absolute path)
#	TMPDIR: The working directory used by Singularity for building the container. This location should be on a Linux-compatible filesystem.
#
# Author: 
#	EM LoCastro June 2020
#
#

SINGRECIPE="${1}"
TMPDIR="${2}"


if [ -z "${SINGRECIPE}" ]
then
	echo "******************************************************************"
	echo "Usage:"
	echo "	sudo ./build_cmd.sh <path-to-sing-def> <path-to-tmpdir-optional>"
	echo "******************************************************************"
	exit
fi


if [ -z "${TMPDIR}" ]
then
	TMPDIR="/singularity"
fi


#get repo name for SIF filename
REPOREMOTE=`git config --get remote.origin.url`
REPONAME=`basename -s .git ${REPOREMOTE}`


#get repo commit hash to stamp output SIF filename
IDSTAMP=`git log | grep commit | head -1 | awk '{ print $2 }' | cut -c1-5`


#full SIF filename
SIFOUT="${REPONAME}_`basename ${SINGRECIPE}`_${IDSTAMP}.sif"


#build the container
singularity build --tmpdir ${TMPDIR} ${SIFOUT} ${SINGRECIPE}


#set container file to be executable
chmod 775 ${SIFOUT}
