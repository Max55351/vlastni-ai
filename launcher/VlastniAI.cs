// Spoustec "Vlastní AI.exe" - spusti program.pyw v Python prostredi projektu.
// Musi lezet ve slozce projektu vedle .venv a program.pyw.
// Preklad: launcher\build.cmd (vestaveny prekladac C# z Windows, nic se nestahuje)
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("Vlastní AI")]
[assembly: AssemblyProduct("Vlastní AI")]
[assembly: AssemblyDescription("Psaní s vlastním jazykovým modelem")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

static class Spoustec
{
    [STAThread]
    static int Main()
    {
        string slozka = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(slozka, ".venv", "Scripts", "pythonw.exe");
        string program = Path.Combine(slozka, "program.pyw");

        if (!File.Exists(python) || !File.Exists(program))
        {
            MessageBox.Show(
                "Nenašel jsem Python prostředí nebo program.\n\n" +
                "Tenhle soubor musí ležet ve složce projektu vedle .venv a program.pyw.\n\n" +
                "Hledal jsem:\n" + python + "\n" + program,
                "Vlastní AI", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        try
        {
            Process.Start(new ProcessStartInfo(python, "\"" + program + "\"")
            {
                WorkingDirectory = slozka,
                UseShellExecute = false,
            });
            return 0;
        }
        catch (Exception e)
        {
            MessageBox.Show("Program se nepodařilo spustit:\n\n" + e.Message,
                "Vlastní AI", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 2;
        }
    }
}
