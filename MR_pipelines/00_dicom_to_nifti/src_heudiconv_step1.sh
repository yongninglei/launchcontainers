# """
# MIT License

# Copyright (c) 2024-2025 Yongning Lei

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# """


# This is the code for the first step of heudiconv
# you can run this one by itself by uncomment the variables
# or you can also use the qsub code to run them by inputing variables


module load apptainer/latest
echo "Now the singularity is loaded, it is: "
module list


echo "Subject: ${sub} "
echo "Session: ${ses} "
cmd="singularity run --cleanenv --no-home --containall \
        	--bind ${basedir}:/base \
	    	--bind /bcbl:/bcbl \
			--bind /export:/export \
        	${sing_path} \
			-d ${dcm_dir}/sub-{subject}/ses-{session}/* \
	    	-s ${sub} \
			-ss ${ses} \
			-o ${outputdir} \
	    	-f convertall \
	    	-c none \
        	-g all \
        	--overwrite "
			# -ss ${ses} \
echo $cmd
eval $cmd

module unload apptainer


# I added this 			-d /base/${dicom_dirname}/sub-{subject}/ses-{session}/*/*.dcm \ is because some of the directory will be read and being processed
# This is the old one works for VOTCLOC/dicom -d /base/${dicom_dirname}/sub-{subject}/ses-{session}/*/*.dcm \
