#define MyAppName "LYKENOX Voice Engine"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "LYKENOX"
#define MyAppExeName "LYKENOX.exe"
#define RepoRoot "..\.."

[Setup]
AppId={{8D6E5B32-9C8B-4E8B-A4F4-3B5A4D8E91F2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\LYKENOX Voice Engine
DefaultGroupName=LYKENOX Voice Engine
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=output
OutputBaseFilename=LYKENOX-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupLogging=yes

[Files]
Source: "{#RepoRoot}\dist\LYKENOX\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Abrir {#MyAppName}"; Flags: nowait postinstall skipifsilent
