# How to set environment variables

nnU-Net requires some environment variables so that it always knows where the raw data, preprocessed data and trained 
models are. Depending on the operating system, these environment variables need to be set in different ways.

Variables can either be set permanently (recommended!) or you can decide to set them everytime you call nnU-Net. 

# Linux & MacOS

## Permanent
Locate the `.bashrc` file in your home folder and add the following lines to the bottom:

```bash
export UnnUNet_raw="/media/fabian/UnnUNet_raw"
export UnnUNet_preprocessed="/media/fabian/UnnUNet_preprocessed"
export UnnUNet_results="/media/fabian/UnnUNet_results"
```

(of course you need to adapt the paths to the actual folders you intend to use).
If you are using a different shell, such as zsh, you will need to find the correct script for it. For zsh this is `.zshrc`.

## Temporary
Just execute the following lines whenever you run nnU-Net:
```bash
export UnnUNet_raw="/media/fabian/UnnUNet_raw"
export UnnUNet_preprocessed="/media/fabian/UnnUNet_preprocessed"
export UnnUNet_results="/media/fabian/UnnUNet_results"
```
(of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your terminal! They will also only apply to the current 
terminal window and DO NOT transfer to other terminals!

Alternatively you can also just prefix them to your nnU-Net commands:

`UnnUNet_results="/media/fabian/UnnUNet_results" UnnUNet_preprocessed="/media/fabian/UnnUNet_preprocessed" UnnUNetv2_train[...]`

## Verify that environment parameters are set
You can always execute `echo ${UnnUNet_raw}` etc to print the environment variables. This will return an empty string if 
they were not set.

# Windows
Useful links:
- [https://www3.ntu.edu.sg](https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html#:~:text=To%20set%20(or%20change)%20a,it%20to%20an%20empty%20string.)
- [https://phoenixnap.com](https://phoenixnap.com/kb/windows-set-environment-variable)

## Permanent
See `Set Environment Variable in Windows via GUI` [here](https://phoenixnap.com/kb/windows-set-environment-variable). 
Or read about setx (command prompt).

## Temporary
Just execute the following before you run nnU-Net:

(powershell)
```powershell
$Env:UnnUNet_raw = "/media/fabian/UnnUNet_raw"
$Env:UnnUNet_preprocessed = "/media/fabian/UnnUNet_preprocessed"
$Env:UnnUNet_results = "/media/fabian/UnnUNet_results"
```

(command prompt)
```commandline
set UnnUNet_raw="/media/fabian/UnnUNet_raw"
set UnnUNet_preprocessed="/media/fabian/UnnUNet_preprocessed"
set UnnUNet_results="/media/fabian/UnnUNet_results"
```

(of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your session! They will also only apply to the current 
window and DO NOT transfer to other sessions!

## Verify that environment parameters are set
Printing in Windows works differently depending on the environment you are in:

powershell: `echo $Env:[variable_name]`

command prompt: `echo %[variable_name]%`
