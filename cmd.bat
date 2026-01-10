@echo off
powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k cd /d %~dp0'"