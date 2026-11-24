; Установщик VCMX2STP (Inno Setup 6).
; Сборка: iscc /DMyAppVersion=1.2.3 /DSourceDir=..\dist\VCMX2STP installer\vcmx2stp.iss
;
; AppId неизменен — новая версия устанавливается поверх старой (обновление),
; настройки пользователя (%APPDATA%\VCMX2STP) сохраняются.
; Программа обновляет себя сама: скачивает VCMX2STP-Setup-<версия>.exe из
; GitHub Releases и запускает его с ключами /SILENT /CLOSEAPPLICATIONS /RELAUNCH.

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\VCMX2STP"
#endif
#ifndef MyAppRepo
  #define MyAppRepo "OWNER/vcmx2stp"
#endif

#define MyAppName "Конвертер VCMX в STEP"
#define MyAppShortName "VCMX2STP"
#define MyAppExeName "VCMX2STP.exe"
#define MyAppPublisher "VCMX2STP"
#define MyAppURL "https://github.com/" + MyAppRepo

[Setup]
AppId={{6F1C2B7E-3A4D-4C59-9E21-8B5A7D3C0F42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
VersionInfoVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppShortName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Установка без прав администратора (в %LOCALAPPDATA%\Programs) — обновление без UAC.
; При запуске от администратора можно выбрать установку для всех пользователей.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
UsePreviousAppDir=yes
UsePreviousPrivileges=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir=..\dist\installer
OutputBaseFilename=VCMX2STP-Setup-{#MyAppVersion}
SetupIconFile=..\src\vcmx2stp\assets\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
; сжатие в отдельном процессе (ultra64 со словарём ~1,5 ГБ вызывал Out of memory)
LZMAUseSeparateProcess=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=yes
ChangesAssociations=yes
ChangesEnvironment=no

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "contextmenu"; Description: "Пункт «Конвертировать в STEP» в контекстном меню файлов .vcmx"; GroupDescription: "Интеграция с Проводником:"

[InstallDelete]
; удаляем файлы прошлой версии, чтобы не оставались устаревшие DLL
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\Удалить {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Контекстное меню для .vcmx (не меняет программу по умолчанию — KUKA.Sim остаётся владельцем расширения)
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.vcmx\shell\VCMX2STP"; ValueType: string; ValueName: ""; ValueData: "Конвертировать в STEP (VCMX2STP)"; Flags: uninsdeletekey; Tasks: contextmenu
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.vcmx\shell\VCMX2STP"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: contextmenu
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.vcmx\shell\VCMX2STP\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: contextmenu

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
; после тихого обновления из программы — запустить её снова
Filename: "{app}\{#MyAppExeName}"; Flags: nowait runasoriginaluser; Check: ShouldRelaunch

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Code]
{ Перезапуск программы после тихого обновления, запущенного из самой программы (ключ /RELAUNCH). }
function ShouldRelaunch: Boolean;
begin
  Result := WizardSilent and (Pos('/RELAUNCH', UpperCase(GetCmdTail)) > 0);
end;
