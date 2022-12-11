pipeline {
    agent { label 'jenkinsfile' }
    triggers {
        pollSCM('H/10 * * * *')
    }
    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }
    stages {
        stage ('build') {
            steps {
                sh '''#!/bin/bash -el
                    # The -x flags indicates to echo all commands, thus knowing exactly what is being executed.
                    # The -e flags indicates to halt on error, so no more processing of this script will be done
                    # if any command exits with value other than 0 (zero)

module purge
module load ecrc-extras
module load mkl/2020.0.166
module load gcc/10.2.0 
module load  cmake/3.19.2  


BASE=$WORKSPACE

## HCORE
cd $BASE
IDIR=$PWD/build/installdir
mkdir -p "${IDIR}"
cd "${IDIR}"/..
rm -rf ./CMake*
cmake .. -DCMAKE_INSTALL_PREFIX=$IDIR
make
make install
#make test
ctest -V
make package
                '''
                archiveArtifacts allowEmptyArchive: true, artifacts: 'build/hcore*.tar.gz'
            }
        }
    }
    // Post build actions
    post {
        //always {
        //}
        //success {
        //}
        //unstable {
        //}
        //failure {
        //}
        unstable {
            emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build is UNSTABLE", recipientProviders: [culprits(),requestor()]
        }
        failure {
            emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build FAILED", recipientProviders: [culprits(),requestor()]
        }
        success {
            emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build OK", recipientProviders: [culprits(),requestor()]
            //build '../al4san-dev/master'
        }
    }
}

