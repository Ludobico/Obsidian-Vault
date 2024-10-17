
## ls, 파일 및 디렉터리 목록 확인

*ls* 는 파일이나 디렉터리 목록을 확인하는 명령어로 list의 약자입니다.
- -l : 상세정보 출력
- -a : 숨김파일을 포함하여 파일을 출력

```bash
┌──(kali㉿kali)-[~]
└─$ ls
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls -l
total 32
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Desktop
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Documents
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Downloads
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Music
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Pictures
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Public
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Templates
drwxr-xr-x 2 kali kali 4096 Oct 17 16:48 Videos
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls -al
total 124
drwx------ 15 kali kali  4096 Oct 17 16:53 .
drwxr-xr-x  3 root root  4096 Oct 17 16:47 ..
-rw-------  1 kali kali     0 Oct 17 16:48 .ICEauthority
-rw-------  1 kali kali    49 Oct 17 16:48 .Xauthority
-rw-r--r--  1 kali kali   220 Oct 17 16:47 .bash_logout
-rw-r--r--  1 kali kali  5551 Oct 17 16:47 .bashrc
-rw-r--r--  1 kali kali  3526 Oct 17 16:47 .bashrc.original
drwxrwxr-x  7 kali kali  4096 Oct 17 16:51 .cache
drwxr-xr-x 12 kali kali  4096 Oct 17 16:49 .config
-rw-r--r--  1 kali kali    35 Oct 17 16:48 .dmrc
-rw-r--r--  1 kali kali 11759 Oct 17 16:47 .face
lrwxrwxrwx  1 kali kali     5 Oct 17 16:47 .face.icon -> .face
drwx------  3 kali kali  4096 Oct 17 16:48 .gnupg
drwxr-xr-x  3 kali kali  4096 Oct 17 16:47 .java
drwxr-xr-x  4 kali kali  4096 Oct 17 16:48 .local
-rw-r--r--  1 kali kali   807 Oct 17 16:47 .profile
-rw-r--r--  1 kali kali     0 Oct 17 16:49 .sudo_as_admin_successful
drwxr-xr-x  2 kali kali  4096 Oct 17 16:48 Public
drwxr-xr-x  2 kali kali  4096 Oct 17 16:48 Templates
drwxr-xr-x  2 kali kali  4096 Oct 17 16:48 Videos

```

## cd, 디렉터리 이동

**cd** 는 현재 디렉터리에서 다른 디렉터리로 이동할 때 사용하는 명령어로 change directory의 약자입니다.

```bash
┌──(kali㉿kali)-[~]
└─$ cd /tmp 
                                                                                                                               
┌──(kali㉿kali)-[/tmp]
└─$ 

```


## pwd, 현재 위치 확인

**pwd** 는 현재 디렉터리 위치를 나타낼 때 사용하는 명령어로 print working directory의 약자입니다.

```bash
┌──(kali㉿kali)-[/tmp]
└─$ cd ..  
                                                                                                                               
┌──(kali㉿kali)-[/]
└─$ pwd
/

```

## mkdir, 디렉터리 생성

**mkdir**는 새로운 디렉터리를 생성할 때 사용하는 명령어로 make directory의 약자입니다. 기본적으로 현재 위치의 하위 디렉터리로 생성하며, 상위 디렉터리가 생성되어 있지 않을 때 **-p** 옵션을 설정하면 상위 디렉터리도 자동으로 생성합니다.

```bash
┌──(kali㉿kali)-[/]
└─$ cd ~ 
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ sudo mkdir test
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos  test
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ sudo mkdir -p ~/test2/subdir
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos  test  test2
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ cd test2
                                                                                                                               
┌──(kali㉿kali)-[~/test2]
└─$ ls
subdir

```

## rmdir & rm, 디렉터리 및 파일 삭제

디렉터리와 파일을 삭제할때 **rmdir** 과 **rm** 명령어를 사용합니다. rmdir은 remove directory의 약자로 디렉터리를 삭제하는데, 디렉터리를 삭제하려면 "쓰기 권한"이 있어야합니다. 단, root 사용자는 모든 권한을 갖고 있기 때문에 모든 디렉터리를 삭제할 수 있습니다. rm은 디렉터리 뿐만 아니라 파일도 삭제할 수 있는 명령어이므로 사용할 때 주의를 기울여야 합니다.

- -r 하위 디렉터리까지 모두 삭제
- -f 삭제 시 내용을 확인하지 않고 삭제

```bash
┌──(kali㉿kali)-[~/test2]
└─$ cd ..   
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos  test  test2

                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ touch test.txt
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos  test  test.txt  test2
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ sudo rm test.txt            
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos  test  test2
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ sudo rm test    
rm: cannot remove 'test': Is a directory
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ sudo rmdir test

┌──(kali㉿kali)-[~]
└─$ sudo rmdir test2
rmdir: failed to remove 'test2': Directory not empty
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ sudo rm -r test2
                                                                                                                               
┌──(kali㉿kali)-[~]
└─$ ls
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos

```

