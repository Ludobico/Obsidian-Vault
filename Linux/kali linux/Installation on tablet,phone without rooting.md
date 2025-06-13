- [[#설치하기전|설치하기전]]
	- [[#설치하기전#Android version 14 ~|Android version 14 ~]]
	- [[#설치하기전#Android version 12 ~ 13|Android version 12 ~ 13]]
- [[#Installation|Installation]]


## 설치하기전

안드로이드 버전이 12이상인 태블릿&핸드폰에는 백그라운드 프로세스 수를 제한하여 프로세스를 죽이는 <font color="#ffff00">Phantom Process Killer</font> 문제가 발생합니다.

### Android version 14 ~

https://www.youtube.com/watch?v=N5Q5J36wIkc

### Android version 12 ~ 13

1. 디버깅 모드하는법
https://www.asus.com/kr/support/faq/1046846/

2. adb 명령어

https://data-science.tistory.com/117

```bash
adb devices

adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent”

adb shell "/system/bin/device_config put activity_manager max_phantom_processes 2147483647”

adb shell settings put global settings_enable_monitor_phantom_procs false

adb shell "/system/bin/dumpsys activity settings | grep max_phantom_processes”

adb shell "/system/bin/device_config get activity_manager max_phantom_processes”
```


## Installation

https://www.youtube.com/watch?v=l41sVhRIRE0

This comment has 2 parts, read the 2nd comment for the rest of the instructions.

The performance of Kali on Android will vary by version, device, and platform. The device I used in the video was the Motorola ThinkPhone:
    https://www.amazon.com/dp/B0C7SSTPD9/

For loss of internet, read the internet fix under Fixes & Configurations.

If you encounter the missing key error from the Kali April update (https://www.kali.org/releases/ ), execute the following from a terminal (https://www.youtube.com/watch?v=AiXoce3rEXs ):
    sudo wget https://archive.kali.org/archive-key.asc -O /etc/apt/trusted.gpg.d/kali-archive-keyring.asc
    sudo apt update -y
I recommend rebooting Linux before starting up Synaptic. From Synaptic Package Manager, click the Reload button on the top left

===================================
INSTALL USERLAND APP
===================================

From the Play Store, install the UserLAnd app.

From the UserLAnd app:
    3 dot menu from the top right > select Settings
    Default Landing Page > select Sessions

===================================
INSTALL KALI
===================================

From the UserLAnd Apps screen > select Kali
Click OK to enable notification permissions for UserLAnd > click Allow to Allow UserLAnd to send you notifications
For desktop environment, select Minimal > click CONTINUE
For connection type, select Graphical > click CONTINUE
Leave Graphical Settings as is > click CONTINUE

When Kali is installed, make the session protected (prevents accidentally deleting Kali):
    Click 3 dot menu from VNC floating menu > select Disconnect
    Go back into the UserLAnd app.
    From the Sessions screen, right click Kali > select Stop Session
    Right click Kali > select Edit
    Check the Protected: option.
    Click on the back navigation button to return to the Sessions screen.
    Go to the Android apps overview and swipe out the UserLAnd app.
    Open UserLAnd.
    From the Sessions screen, click on Kali to start the Kali session.

The floating menu is called the VNC floating menu:
    To move the menu, hold a left click on the left most icon and drag the menu.
    Toggle Extra Keys by clicking the 2nd icon from the left.
    To disable extra keys, click the 3 dot menu from the VNC floating menu > select Disable Extra Keys

===================================
INSTALL AND START UP XFCE
===================================

To see the top right terminal more clearly:
    Hover mouse over the terminal.
    Do a 2 finger pinch zoom in.

To type text into the terminal, hover the mouse over the terminal window.

Download the Kali software key (all 1 line):
    sudo wget --no-check-certificate https://archive.kali.org/archive-key.asc -O /etc/apt/trusted.gpg.d/kali-archive-keyring.asc

Update:
    sudo apt update -y
    sudo apt dist-upgrade -y

Install software (During the kali-desktop-xfce install, a dialog will ask to select the Keyboard layout. Up/Down arrow keys to scroll, Tab to toggle between OK and Cancel, and Enter to continue):
    sudo apt install nano -y
    sudo apt install dialog -y
    sudo apt install kali-defaults -y
    sudo apt install kali-desktop-xfce -y
    sudo apt install kali-wallpapers-2024 -y
    sudo apt install ffmpeg -y
    sudo apt install synaptic -y
    sudo apt install chromium -y

Prevent XFCE from automatically starting up by moving autostart files into a hold folder (allows switching between desktops, can be moved back if needed):
    mkdir /etc/X11/holdXsession.d
    mv /etc/X11/Xsession.d/* /etc/X11/holdXsession.d

Prevent other sub processes from automatically starting up by moving the autostart files into a hold folder (helps prevent issues and improve performance, can be moved back if needed):
   mkdir /etc/xdg/holdautostart
   mv /etc/xdg/autostart/* /etc/xdg/holdautostart

If ~/.config/autostart exists, move the files into a hold folder (can be moved back if needed):
    mkdir ~/.config/holdautostart
    mv ~/.config/autostart/* ~/.config/holdautstart

Create startup script for XFCE:
    nano /usr/bin/gox
        startxfce4 &> /dev/null &
    Ctrl+O, Enter, and Ctrl+X to save the file and exit nano.

Make the script executable:
    chmod +x /usr/bin/gox

Restart Linux:
    Click on the 3 dot menu from the VNC floating menu > select Disconnect
    Open UserLAnd.
    Right click the Kali session > select Stop Session
    Swipe out the UserLAnd app from Apps Overview.
    Open UserLAnd.
    Click on Kali from the Sessions screen.

From the terminal (if the desktop is too small and pinch to zoom isn't enough, watch the scaling video: https://www.youtube.com/watch?v=uSQS8kZV_UA ):
    gox
    exit

-----------------------------------

PROPER shutdown:
    Click the 3 dot menu in the VNC Floating Menu > select Disconnect
    Go back into the UserLAnd app.
    Right click the Kali session > select Stop Session
    Swipe out UserLAnd app from Apps Overview.

If UserLAnd is improperly shutdown, reboot the Android device.

===================================
FIXES AND CONFIGURATIONS
===================================

Improve XFCE performance:

Set top panel to a solid color:
    Right click top Panel > Panel > select Panel Preferences...
    Appearance tab > click the Style pulldown > select Solid color

Disable display compositing:
    Menu > Click the Settings Manager button from the bottom right of the menu
    Window Manager Tweaks
    Compositor tab > uncheck Enable display compositing

Disable Screensaver and Lock screen:
    Menu > Click the Settings Manager button from the bottom right of the menu
    Xfce Screensaver
    Screensaver tab > toggle off Enable Screensaver
    Lock Screen tab > toggle off Enable Lock Screen 

-----------------------------------

Configure top panel:

Remove Log Out Button (Log Out causes issues):
    Right click Log Out button from the top right > select Remove
    Click the Remove button.

Remove Generic Monitor (Generic Monitor doesn't have access to CPU usage on Android):
    Right click Generic Monitor near the top right (genmon) > select Remove
    Click the Remove button.

Remove Volume Control (not needed because there is no sound for Linux on Android):
    Right click the Volume Control from the top right > select Remove
    Click the Remove button.

Remove Root Terminal Emulator (Root Terminal Emulator causes issues):
    Right click Terminal Emulator near the top left > select Properties
    Select Root Terminal Emulator
    Click the - button
    Click the Remove button.

Set the time zone from a terminal (click Terminal Emulator near the top left to open a terminal. Ctrl+Shift++ and Ctrl+- to temporarily adjust the terminal text size):
    sudo dpkg-reconfigure tzdata
    Press Enter to continue.
    Select Geographic area (Up/Down arrow keys to scroll) > press Enter to continue
    Select Time zone (Up/Down arrow keys to scroll) > press Enter to continue
    Close the terminal, and the clock will update within a moment.

-----------------------------------

Firefox Fix:

Open Firefox and go to the URL about:config.
Click Accept the Risk and Continue button.
Enter sandbox into the search bar.
    For security.sandbox.content.tempDirSuffix, click the far right arrow to clear the value.
    Set any values with a number to 0
    Set any true/false values to false.
Restart Firefox.

-----------------------------------

Chromium Fix:

Menu > Usual Applications > Internet > right click Chromium Web Browser > select Edit Application...
Change the Command to:
    chromium --no-sandbox --test-type --password-store=basic
Click the Save button.

-----------------------------------

Normally, to change the wallpaper, we would right click inside the desktop, select Desktop Settings..., and then change the wallpaper.

If Desktop Settings is broken, make a copy of the original wallpaper:
    cd /usr/share/backgrounds/kali                              
    cp kali-tiles-16x9.jpg keep.jpg
Create the following go script that will set the desired wallpaper to be the desktop background:
    nano /usr/bin/goback

cd /usr/share/backgrounds/kali

if [ -e "$1".png ] && [ ! -e "$1".jpg ] ; then
    sudo ffmpeg -i "$1".png "$1".jpg &> /dev/null
elif [ -e "$1".jpeg ] && [ ! -e "$1".jpg ] ; then
    sudo ffmpeg -i "$1".jpeg "$1".jpg &> /dev/null
fi

if [ -e "$1".jpg ] ; then
    sudo rm kali-tiles-16x9.jpg
    sudo cp "$1".jpg kali-tiles-16x9.jpg
    xfdesktop -Q
    xfdesktop &
else
    echo "Image not found."
fi

Ctrl+O, Enter, and Ctrl+X to save the file and exit nano.

Make the script executable:
    chmod +x /usr/bin/goback

If using a non-Kali wallpaper, copy it to /usr/share/backgrounds/kali. To use the script, execute the goback script with the name of the wallpaper file. For example, to change the wallpaper to be the kali-ferrofluid-16x9.jpg wallpaper:
    goback kali-ferrofluid-16x9

FFmpeg will automatically convert most formats by using the given input and output file extensions (.ext). For example if we wanted to convert my_image.jpg to a .png, execute "ffmpeg -i my_image.jpg my_image.png):
    ffmpeg -i <input-image-name>.ext <output-image-name>.ext
For more info on ffmpeg:
    ffmpeg --help
    man ffmpeg

-----------------------------------

Fix Synaptic:

Create a launch script for synaptic:
    nano /usr/bin/gosyn
        xhost + &&
        sudo synaptic &&
        xhost -
    Ctrl+O, Enter, and Ctrl+X to save the file and exit nano.

Make gosyn script executable:
    chmod +x /usr/bin/gosyn

Change Synaptic launch command to use the gosyn script:
    Menu > Usual Applications > System > right click Synaptic Package Manager > select Edit Application...
    Change Command to:
        gosyn
    Click the Save button.

With Android I recommend searching by Name. Searching by Description and Name will be significantly slower.

-----------------------------------

Read 2nd comment for the rest of the instructions.