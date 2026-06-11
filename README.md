python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env   # fill in printer + telegram values
python -m app

//Requirements
1. Download Bambu Studio and Bambu Network plugin for livestream
2. In Privacy under Settings, give VSCode access to Local Network
3. On Bambu Printer, select "Only LAN Mode" and "Developer Mode" to start prints

I'm using Obico open source ONNX model weights & spaghetti variable values. The only difference is that I pause on any spaghetti detected.

Bambu Labs camera + Obico model - backend

//Features (Bambu Studio device-tab replacement)
- Monitor: live camera, print state, layer/percent progress, ETA, HMS alerts, wifi signal
- Control: pause/resume/stop, chamber light, speed level (silent/standard/sport/ludicrous)
- Temperatures: nozzle + bed read/set; part & aux fan read/set
- Movement: home, XYZ jog (blocked while printing), extrude/retract (nozzle >= 170C)
- Filament: AMS trays + external spool status
- Files: list/print/delete .3mf on printer SD (root + cache) over FTPS
- Slice: upload STL/3MF/STEP/OBJ -> Bambu Studio CLI slices with real BBL presets ->
  time/filament estimate -> print directly or download the gcode-3mf
  (needs Bambu Studio installed; see SLICER_* in .env.example)

//API
GET  /api/health /api/snapshot /api/events /api/detector /api/frame.jpg /api/stream (SSE)
POST /api/print/start (multipart .3mf) | /api/print/{pause|resume|stop}
POST /api/light/{on|off|toggle}
POST /api/temp/{nozzle|bed} {target} | /api/fan/{part|aux|chamber} {percent} | /api/speed {level}
POST /api/move {axis,dist} | /api/move/home | /api/extrude {dist}
GET  /api/files | POST /api/files/print {path,...} | DELETE /api/files?path=
GET  /api/slicer/presets | POST /api/slice (multipart model) |
POST /api/slice/{job}/print | GET /api/slice/{job}/download
